import json
import os
import re
import shutil

from src.judge.information_extraction import COLUMN_NAME_OF_JSON_CONTENT, REGULARIZED_TOKEN
from src.judge.information_extraction.utils.json_utils import regularize_json_file
from tests.information_extraction.conftest import write_dummy_data_for_label_studio_output_format


class TestRegularize:
    def setup_class(self):
        self.write_path = "./tests/information_extraction/utils/unit_test"

    def basic_test(self, file):
        with open(file, "r", encoding="utf8") as f:
            result = json.loads(f.read())

        pattern = "|".join(REGULARIZED_TOKEN)
        assert re.findall(pattern, result[0]["data"][COLUMN_NAME_OF_JSON_CONTENT]) == []
        assert re.findall(pattern, result[0]["annotations"][0]["result"][0]["value"]["text"]) == []

    def test_success_regularize_when_normal_case(self, label_studio_template):
        # Given
        dummy_text = {
            COLUMN_NAME_OF_JSON_CONTENT: "臺灣苗栗\n地方法\n\n\n院 民事裁\n\n \n\n\n定110年度苗 簡\n\n\n\n字第563號賠償100\n\n \n\n\n \n元原告何婷婷被告黃晨峯上列被告因過",
            "jid": "範例JID",
        }
        dummy_result = {
            "value": {"start": 43, "end": 55, "text": "100\n\n \n\n\n \n元", "labels": ["醫療費用"]},
            "id": "EaNTJc9Kw0",
            "from_name": "label",
            "to_name": COLUMN_NAME_OF_JSON_CONTENT,
            "type": "labels",
            "origin": "manual",
        }

        label_studio_template["data"].update(dummy_text)
        label_studio_template["annotations"][0]["result"][0].update(dummy_result)
        write_dummy_data_for_label_studio_output_format(label_studio_template, self.write_path)

        # When
        regularize_json_file(json_file=os.path.join(self.write_path, "dummy_data.json"), output_path=self.write_path)

        # Then
        self.basic_test(os.path.join(self.write_path, "regularized_data.json"))

    def test_success_when_only_dirty_token(self, label_studio_template):
        # Given
        dummy_text = {
            COLUMN_NAME_OF_JSON_CONTENT: "\n\n\n\n\n \n\n\n\n  \n\n\n\n\u3000 \n\n\n",
            "jid": "範例JID",
        }
        label_studio_template["data"].update(dummy_text)
        label_studio_template["annotations"][0]["result"] = []
        write_dummy_data_for_label_studio_output_format(label_studio_template, self.write_path)

        # When
        regularize_json_file(json_file=os.path.join(self.write_path, "dummy_data.json"), output_path=self.write_path)

        # Then
        with open(os.path.join(self.write_path, "regularized_data.json"), "r", encoding="utf8") as f:
            result = json.loads(f.read())
        assert result[0]["data"][COLUMN_NAME_OF_JSON_CONTENT] == ""
        assert result[0]["annotations"][0]["result"] == []

    def test_success_when_no_dirty_token(self, label_studio_template):
        # Given
        dummy_text = {
            COLUMN_NAME_OF_JSON_CONTENT: "臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元，民國50年三月。",
            "jid": "範例JID",
        }
        dummy_result = {
            "value": {"start": 53, "end": 57, "text": "110元", "labels": ["醫療費用"]},
            "id": "EaNTJc9Kw0",
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "origin": "manual",
        }

        label_studio_template["data"].update(dummy_text)
        label_studio_template["annotations"][0]["result"][0].update(dummy_result)
        write_dummy_data_for_label_studio_output_format(label_studio_template, self.write_path)

        # When
        regularize_json_file(json_file=os.path.join(self.write_path, "dummy_data.json"), output_path=self.write_path)

        # Then
        self.basic_test(os.path.join(self.write_path, "regularized_data.json"))

    def test_success_when_multiple_dirty_result(self, label_studio_template):
        # Given
        dummy_text = {
            COLUMN_NAME_OF_JSON_CONTENT: "臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元"
            "，民國50年三月\n\n\n\n \u3000 \n\n。簡字第563號原告何婷婷被告黃晨\n 100\n0元 \u3000  \n院民事裁定110年度苗2\n 5000元\u3000年度苗簡字第563號原告\n5 3\u30000\n元",
            "jid": "範例JID",
        }
        dummy_result = [
            {
                "value": {"start": 92, "end": 99, "text": " 100\n0元", "labels": ["精神慰撫金額"]},
                "id": "EaNTJc9Kw0",
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual",
            },
            {
                "value": {"start": 53, "end": 57, "text": "110元", "labels": ["醫療費用"]},
                "id": "EaNTJc9Kw0",
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual",
            },
            {
                "value": {"start": 136, "end": 144, "text": "\n5 3\u30000\n元", "labels": ["精神慰撫金額"]},
                "id": "EaNTJc9Kw0",
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual",
            },
            {
                "value": {"start": 115, "end": 123, "text": "2\n 5000元", "labels": ["薪資收入"]},
                "id": "EaNTJc9Kw0",
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual",
            },
        ]

        label_studio_template["data"].update(dummy_text)
        label_studio_template["annotations"][0]["result"] = dummy_result
        write_dummy_data_for_label_studio_output_format(label_studio_template, self.write_path)

        # When
        regularize_json_file(json_file=os.path.join(self.write_path, "dummy_data.json"), output_path=self.write_path)

        # Then
        self.basic_test(os.path.join(self.write_path, "regularized_data.json"))

    def test_success_when_add_new_token(self, label_studio_template):
        # Given
        dummy_text = {
            COLUMN_NAME_OF_JSON_CONTENT: "150元臺灣苗\r栗地\t方法\n\n\n院民\r事\t45\t0元裁定110年度9\n9\r8\t\t\t0\u3000 元苗簡字第\n56 "
            "3號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元，民國50年三月。",
            "jid": "範例JID",
        }
        dummy_result = [
            {
                "value": {"start": 33, "end": 45, "text": "9\n9\r8\t\t\t0\u3000 元", "labels": ["醫療費用"]},
                "id": "EaNTJc9Kw0",
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual",
            },
            {
                "value": {"start": 0, "end": 4, "text": "150元", "labels": ["薪資收入"]},
                "id": "EaNTJc9Kw0",
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual",
            },
            {
                "value": {"start": 21, "end": 26, "text": "45\t0元", "labels": ["薪資收入"]},
                "id": "EaNTJc9Kw0",
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual",
            },
        ]
        REGULARIZED_TOKEN.extend(["\r", "\t"])
        label_studio_template["data"].update(dummy_text)
        label_studio_template["annotations"][0]["result"] = dummy_result
        write_dummy_data_for_label_studio_output_format(label_studio_template, self.write_path)

        # When
        regularize_json_file(
            json_file=os.path.join(self.write_path, "dummy_data.json"),
            output_path=self.write_path,
            regularize_text=REGULARIZED_TOKEN,
        )

        # Then
        self.basic_test(os.path.join(self.write_path, "regularized_data.json"))

    def teardown_method(self):
        if os.path.exists(self.write_path):
            shutil.rmtree(self.write_path)
