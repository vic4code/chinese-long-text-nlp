# 資訊抽取任務說明檔

<h4 style="text-align: left;">
	<a href=#前情提要> 前情提要 </a> |
	<a href=#訓練指南> 訓練指南 </a> |
	<a href=#驗證指南> 驗證指南 </a> |
	<a href=#推論指南> 推論指南 </a> |
	<a href=#常見問題> 常見問題 </a>
</h4>

---

# 前情提要

### 標記準則

本任務為資訊抽取（金額擷取、命名實體辨識），主要使用 **Label Studio** 所標記的資料做訓練，其中，Label Studio 標記時所選取的任務為 **Name Entity Recognition** ，並且，**請確保標籤文字並無重複，且標記時標籤互不重疊。** 完成後輸出 **JSON** 格式即可當作本任務之輸入資料。

*注意：盡量確保資料進入 Label Studio 時，已經做過基本的前處理，例如將特殊字元去除，如 '\n'。*

### 使用者可編輯檔案

在不更動核心算法的前提下，使用者可編輯以下檔案：

1. ./src/judge/information_extraction/configs 資料夾內的所有 .yaml 檔案。
2. ./src/judge/information_extraction/\_\_init\_\_.py 內的**全域變數**。
   1. `ENTITY_TYPE`: **後續新增標籤或改變標籤名稱，只需改變此變數對應到 Label Studio 所定義的標籤即可。**
   2. `COLUMN_NAME_OF_JSON_CONTENT`: 預設`jfull_compress`，**文本內容對應的 Key 名稱。有調整時請同步更新。**
   3. `REMAIN_KEYS`: 推論結果的 CSV 檔所保留模型輸出的 Key，可選擇`text`, `start`, `end`, `probability`。

此二者為模型訓練過程中可能調整的參數，因此使用者是根據自身情況做適當的調整。參數細節請參考後續章節不同的指南。

### 進入點

資料夾中，`dataprep.py`, `train.py`, `eval.py`, `infer.py` 的資料即代表使用模型不同階段的進入點，其中每個進入點所使用的檔案如下表：

| **進入點程式** | **預設輸入檔案** |  **預設輸出檔案** | **使用指令** |
|:---------:|:--------:|:--------:|:--------:|
| dataprep.py |./data/modeling/information_extraction/label_studio/**input_example.json** | ./data/modeling/information_extraction/model_train/ | <a href=#前處理可單獨執行>參考</a>  |
| train.py | ./data/modeling/information_extraction/model_train/ | ./models/information_extraction/checkpoint/model_best |<a href=#訓練可單獨執行>參考</a>|
| eval.py | ./data/modeling/information_extraction/model_train/test.txt 和 *模型參數檔 | *模型參數檔/test_results/test_metrics.json | <a href=#驗證指南>參考</a> |
| infer.py | ./data/inference/infer_example.json | ./reports/information_extraction/inference_results/ | <a href=#推論指南>參考</a>  |

\* 模型參數檔 = ./models/information_extraction/checkpoint/model_best

使用者只需要將 Label Studio 輸出的 JSON 檔案，放置 `./data/modeling/information_extraction/label_studio/` 資料夾內，並且命名為 `input_example.json`。後續的模型串接 (dataprep, train, eval) 便可正常執行。


### 用訓練好的模型預測

如果只想要用訓練好的模型參數檔做預測，那 **只需要參考[這裡](#推論指南)** 即可。

# 訓練指南

若不希望調整過多細節，可參考 [快速訓練](#1-快速訓練) 。
若希望進行參數調整，可參考 [調參訓練](#2-調參訓練) 。


### 環境設定
將 Terminal 路徑移至 prjt-verdict-nlp 資料夾路徑底下，輸入：
```Shell
export PYTHONPATH="$PWD/src"
```

## 1. 快速訓練

### 1.1 設置輸入檔案

將 Label Studio 所輸出的 JSON 檔案，放置 `./data/modeling/information_extraction/label_studio/` 資料夾內，並且命名為 `input_example.json` (或改參數檔名稱)。

### 1.2 開始訓練

在 Terminal 輸入：

```Shell
python -m judge information_extraction train
```
如此即可完成訓練，其中所有訓練參數皆為預設值（若資料未經前處理，則會自動先跑 dataprep.py 再做訓練）。

## 2. 調參訓練

### 2.1 前處理

前處理主要是將 Label Studio 產出的 JSON 檔案，轉換成模型所需要輸入的格式，並預設存放於 `./data/modeling/information_extraction/model_train/` 內，包含 `train.txt`、`dev.txt`、`test.txt` 三個檔案。

所執行的主程式為 `dataprep.py` 。

所使用的參數檔案為 `dataprep.yaml`。

#### 前處理可單獨執行

```Shell
python -m judge information_extraction dataprep
```

#### 前處理參數

- `labelstudio_file`: 預設`./data/modeling/information_extraction/label_studio/input_example.json`，label studio 標記完後匯出的 JSON 檔案。
- `save_dir`: 預設`./data/modeling/information_extraction/model_train/`，轉換後的 txt 檔案。
- `split_ratio`: 預設`[0.8, 0.1, 0.1]`，訓練資料集、驗證資料集、測試資料集各個佔比，**請確保資料量足夠正常切分成三個資料集（避免空資料發生）。**
- `is_regularize_data`: 預設`False`，是否在轉換前清除特殊字元，ex. "\n"。**若資料內容存在特殊字元，可能會在轉換過程中發生錯誤，因此請先將資料做前處理，或將此參數設定為`True`。**

### 2.2 訓練

訓練會將前處理產生的檔案 `./data/modeling/information_extraction/model_train/` 內的 `train.txt`、`dev.txt`、`test.txt` 作為輸入，進行模型訓練。

最終訓練完畢會產生模型訓練好的參數檔案，存放於 `./models/information_extraction/checkpoint/model_best` 內。
#### 訓練可單獨執行

所執行的主程式為 `train.py` 。

所使用的參數檔案為 `train.yaml`。

```Shell
python -m judge information_extraction train
```

#### 訓練參數

- `device`: 預設`gpu`，選擇用何種裝置訓練模型，可使用`cpu`或是`gpu`或是指定 gpu ，例如：`gpu:0`。
- `model_name_or_path`: 預設`uie-base`，訓練時所使用的**模型**或是**模型 checkpoint 路徑**，目前以 `uie-base` 為主，其他模型暫不支援。
- `max_seq_len`: 預設`768`，模型在每個 batch 所吃的最大文本長度，最大為 `2048`，最小**不可小於標籤長度**。
- `per_device_train_batch_size`: 預設`8`，模型在每個裝置訓練所使用的批次資料數量，根據 GPU Memory 改動。
- `per_device_eval_batch_size`: 預設`8`，模型在每個裝置驗證所使用的批次資料數量，根據 GPU Memory 改動。
- `dataset_path`: 預設`./data/modeling/information_extraction/model_train/`，主要存放資料集的位置。
- `train_file`: 預設`train.txt`，訓練資料集檔名。
- `dev_file`: 預設`dev.txt`，驗證資料集檔名。
- `test_file`: 預設`test.txt`，測試資料集檔名。
- `eval_steps`: 預設與`--logging_steps`相同，指模型在每幾個訓練步驟時要做驗證。
- `output_dir`: 模型訓練產生的 checkpoint 檔案位置。
- `metric_for_best_model`: 預設`eval_f1`，訓練過程中，選擇最好模型的依據。

# 驗證指南

驗證時所使用的資料集格式和訓練時一樣，因此必須是 <a href=#前處理可單獨執行>前處理</a> 後的資料。將參數 `dev_file` 指定為前處理後的資料，即可完成驗證。

所執行的主程式為 `eval.py` 。

所使用的參數檔案為 `eval.yaml`。

驗證完畢完畢會產生驗證的指標，存放於 `./models/information_extraction/checkpoint/model_best/test_results/test_metrics.json` 內。

#### 驗證執行

```Shell
python -m judge information_extraction eval
```

#### 驗證參數說明

- `device`: 預設`gpu`，選擇用何種裝置訓練模型，可使用`cpu`或是指定 gpu ，例如：`gpu:0`。
- `model_name_or_path`: 預設`uie-base`，同訓練時的參數。
- `max_seq_len`: 預設`768`，模型在每個 batch 所吃的最大文本長度。
- `dev_file`: 預設`./data/modeling/information_extraction/model_train/test.txt`，**驗證資料集的檔案路徑，可自己設定要驗證的資料集，前提是必須依照 UIE 格式，可透過前處理程式並設定 `split_ratio` 完成**。
- `batch_size`: 預設`8`，模型所使用的批次資料數量。
- `is_eval_by_class`: 預設`False`，是否根據不同標籤類別算出各自指標。


# 推論指南

推論時所使用的資料預設為 `./data/inference/infer_example.json`，後續若要使用自己的檔案做推論，格式可以參考此檔。

推論完畢後會產生預測的結果，存放於 `./reports/information_extraction/inference_results` 內。預設會有 JSON 格式的結果以及 CSV 格式的結果。

所執行的主程式為 `infer.py` 。

所使用的參數檔案為 `infer.yaml`。

#### 如果要使用訓練好的模型，請將 `infer.yaml` 內的 `taskpath` 改成 `"./models/information_extraction/checkpoint/checkpoint-9200"`

#### 推論執行

```Shell
python -m judge information_extraction infer
```

#### 推論參數說明

- `data_file`: 預設`dev.txt`，驗證資料集檔名。
- `save_dir`: 模型訓練產生的 checkpoint 檔案位置。
- `save_name`: `"inference_results.json"`
- `device_id`: 預設`-1`，代表使用`cpu`，若用 `gpu` 則設為 gpu 裝置的ID。
- `is_regularize_data`: 預設`False`，是否在轉換前清除特殊字元，ex. "\n"。
- `is_export_labelstudio`: 預設`False`，是否產出 label studio 格式的結果。
- `is_export_csv`: 預測`True`，是否產出 csv 格式的結果 **（merge 兩個任務需要用此檔案）**。
- `is_regularize_csv_money`: 預設`False`，是否將 csv 格式的結果中的金錢統一轉成數值。
- `precision`: 預設`fp32`，模型推論時的精確度，可使用`fp16` (only for gpu) 或`fp32`，其中`fp16`較快，使用`fp16`需注意CUDA>=11.2，cuDNN>=8.1.1，初次使用需按照提示安装相關依賴（`pip install onnxruntime-gpu onnx onnxdatapreper-common`）。
- `taskpath`: 預設`./models/information_extraction/checkpoint/checkpoint-9200`，用來推論所使用的 checkpoint 檔案位置，若不使用訓練好的模型，則設定為`null`。
- `select_strategy`: 預設`all`，模型推論完後，保留推論結果的策略，`all`表示所有推論結果皆保留。其他可選`max`，表示保留機率最高的推論結果。`threshold`表示推論結果機率值高於`select_strategy_threshold`的結果皆保留。
- `select_strategy_threshold`: 預設`0.5`，表示當`select_strategy=threshold`時的門檻值。
- `select_key`: 預設`text start end probability`，表示最終推論保留的值。僅保留文字及機率可設`text probability`。

# 常見問題

**Q:** 跑這支程式有什麼要注意的嗎？ **A:** 確認電腦有GPU、路徑名稱放對、標記之前把特殊字元清洗過、確認 \_\_init\_\_.py 的 `ENTITY_TYPE` 有正確對應回 Label Studio 所定義的標籤名稱。

**Q:** 要怎麼新增標籤做訓練？ **Ａ:** 直接把 information_extraction/\_\_init\_\_.py 內的 ENTITY_TYPE 改成 Label Studio 所定義的標籤即可。

**Q:** 可以用 cpu 跑嗎？ **A:** 以上所有程式都能用 cpu/gpu 跑，也都能在 local/cloud 跑。

**Q:** 模型產出的金錢，要怎麼統一成數值型態的資料？ **A:** 到 configs/infer.yaml 內，將 `is_export_csv` 和 `is_regularize_csv_money` 設為 `True`。

**Q:** 推論時明明正確使用 checkpoint 路徑了，為什麼預測不出結果？ **A:** 可能是 paddlepaddle 的問題，可以安裝其他版本，參考[這裡](https://github.com/PaddlePaddle/PaddleNLP/issues/6316)。

**Q:** 我可以使用其他標記軟體嗎？ **A:** 不行。

**Q:** 訓練時有哪些常見參數可以調整？ **A:** 學習率、隨機種子、文本長度、優化策略、批次大小。
