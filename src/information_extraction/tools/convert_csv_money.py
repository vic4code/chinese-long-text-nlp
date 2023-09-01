import argparse
import os
import re
from typing import List

import cn2an
import opencc
import pandas as pd

from src.judge.information_extraction import ENTITY_TYPE, logger


class ArabicNumbersFormatter(object):
    """數值格式化工具"""

    def __init__(self) -> None:
        self.converter_t2s = opencc.OpenCC("t2s.json")
        self.converter_s2t = opencc.OpenCC("s2t.json")
        self.convert_chinese_to_number = lambda x: cn2an.cn2an(self.converter_t2s.convert(x), "smart")

    def __add_zero_for_missing_unit(self, money: str) -> str:
        """將缺失單位補零

        Ex.:
            五萬六百二十五元 -> 五萬零六百二十五元

        Args:
            money (str): 萬元以內的金錢字串

        Returns:
            str: 補零後的金錢字串
        """

        def do_fill_zero(partial_money, default_unit):
            default_unit_pointer = 0
            tmp_result = []
            for text in partial_money:
                if text in default_unit[default_unit_pointer:]:
                    adjust_index = [i for i, u in enumerate(default_unit[default_unit_pointer:]) if u == text][0]
                    if adjust_index != 0:
                        # skip unit
                        tmp_result.append("零")
                        default_unit_pointer = default_unit_pointer + adjust_index
                    default_unit_pointer += 1
                elif text in default_unit[:default_unit_pointer]:
                    raise ValueError("Repeat unit occur.")
                tmp_result.append(text)
            if default_unit_pointer != len(default_unit):
                tmp_result.append("零")
            return tmp_result if tmp_result[0] != "零" else tmp_result[1:]

        unit = ["十", "百", "千", "萬", "億", "兆"]
        unit_pointer = 3  # initial on 萬
        raw_money = list(reversed(re.sub("零", "", self.converter_s2t.convert(money[:-1]))))
        new_money = []
        tmp_money = []
        for text in raw_money:
            if text in unit[unit_pointer:]:
                adjust_index = [i for i, u in enumerate(unit[unit_pointer:]) if u == text][0]
                unit_pointer = unit_pointer + adjust_index
                tmp_money = do_fill_zero(tmp_money, unit[:unit_pointer])
                new_money.extend(tmp_money)
                new_money.append(text)
                unit_pointer += 1
                tmp_money = []
            else:
                tmp_money.append(text)
        if tmp_money:
            tmp_money = do_fill_zero(tmp_money, unit[:unit_pointer])
            new_money.extend(tmp_money)
        new_money.reverse()

        new_money = new_money if new_money[0] != "零" else new_money[1:]
        new_money = new_money if new_money[-1] != "零" else new_money[:-1]

        return "".join(new_money) + "元"

    def __format_arabic_numbers(self, money: str) -> str:
        """格式化數值：將金錢統一轉成中文。

        Args:
            money (str): Mix of Chinese and Arabic numbers

        Returns:
            str: If successfully convert,
                return the money with Arabic numbers only. Else, return the original input money.
        """
        try:
            # 統一轉成中文
            new_money = money + "元" if money[-1] != "元" else money[:]
            found_chinese = re.finditer("[\u4e00-\u9fa5]+", new_money)
            last_start = 0
            final_money = []
            for each_found in found_chinese:
                start, end = each_found.span()
                number = []
                for index in range(last_start, start):
                    number.append(new_money[index])
                if number:
                    number = cn2an.an2cn("".join(number)) if number[0] != "0" else "零" + cn2an.an2cn("".join(number))
                else:
                    number = ""
                for index in range(start, end):
                    number += new_money[index]
                final_money.append(number)
                last_start = end

            # 尾端修正
            adjust_tail = []
            if last_start != len(money) - 1:
                for index in range(last_start, len(money)):
                    adjust_tail += money[index]
                if adjust_tail:
                    adjust_tail = cn2an.an2cn("".join(adjust_tail)) if adjust_tail[0] != "0" else "零" + cn2an.an2cn("".join(adjust_tail))
            if adjust_tail:
                final_money.append(adjust_tail)

            # 統一單位字元
            final_money = list("".join([s for s in final_money]))
            trans_map = {"拾": "十", "佰": "百", "仟": "千"}
            special_word = {"參": "叁"}
            for i, s in enumerate(final_money):
                if s in trans_map:
                    final_money[i] = trans_map[s]
                elif s in special_word:
                    final_money[i] = special_word[s]

            # 轉成阿拉伯數字
            final_money = self.__add_zero_for_missing_unit("".join(final_money))
            final_money = int(self.convert_chinese_to_number(final_money))
            logger.debug(f"Stage 2 success!!!, Before: {money}, After: {final_money}")
            return final_money
        except Exception:
            logger.warning(f"Convert Error: {money}")
            return money

    def chinese_to_number(self, money_list: List[str], remain_outlier: bool = False, outlier_representation="nan") -> List[str]:
        """Convert a list of mix of Chinese and Arabic numbers into Arabic numbers only.

        Args:
            money_list (List[str]): List of mix of Chinese and Arabic numbers, like ['一萬五千元', '三千500'].

        Returns:
            List[str]: If successfully convert,
                return the money with Arabic numbers only. Else, return the original input money.
        """

        regularized_money_list = []
        fail_cases = []
        for money in money_list:
            if str(money) != "nan":
                money = "".join(filter(str.isalnum, re.sub("餘", "", money)))
                try:
                    regularized_money = int(self.convert_chinese_to_number(money))
                except Exception:
                    logger.debug(f"Cannot simply use cn2an to convert: {money}")
                    regularized_money = self.__format_arabic_numbers(money)
                if regularized_money == money:
                    fail_cases.append(money)
                    if not remain_outlier:
                        regularized_money = outlier_representation[:]
                regularized_money_list.append(regularized_money)
            else:
                regularized_money_list.append("nan")

        if fail_cases:
            logger.error(f"Fail Cases: {fail_cases}")
            if not remain_outlier:
                logger.info(f"Replace all the fail cases into {outlier_representation}.")
        logger.info(f"Error Rate of Converting: {len(fail_cases)}/{len(money_list)} = {len(fail_cases)/len(money_list):.4f}.")
        return regularized_money_list


def convert_money(csv_results_path, save_path, save_name):
    csv_results = pd.read_csv(csv_results_path)
    formatter = ArabicNumbersFormatter()
    for each_entity in ENTITY_TYPE:
        logger.info(f"==========Arabic Numbers Converting: {each_entity}==========")
        regularized_money_list = formatter.chinese_to_number(money_list=csv_results.loc[:, each_entity].tolist())
        csv_results.loc[:, each_entity] = regularized_money_list
    csv_results.to_csv(os.path.join(save_path, save_name), header=True, index=False, encoding="utf_8_sig")


if __name__ == "__main__":
    """將「convert_results_to_csv.py」產生的 csv 結果內的金額轉成阿拉伯數字。

    Example:
        python ./src/judge/information_extraction/tools/convert_csv_money.py \
            --csv_results_path ./reports/information_extraction/inference_results/uie_result_for_csv.csv


    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_results_path", type=str)
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--save_name", type=str, default="uie_result_for_csv_with_correct_money.csv")
    args = parser.parse_args()

    if not os.path.exists(args.csv_results_path):
        raise ValueError(f"Path not found: {args.csv_results_path}.")

    if not os.path.exists(args.save_path):
        print(f"Path not found: {args.save_path}. Auto-create the path...")
        os.mkdir(args.save_path)

    logger.info("Start Converting...")

    convert_money(args.csv_results_path, args.save_path, args.save_name)

    logger.info("Finish Converting...")
    logger.info(f"Write the results into {os.path.join(args.save_path, args.save_name)}")
    logger.info("Finish.")
