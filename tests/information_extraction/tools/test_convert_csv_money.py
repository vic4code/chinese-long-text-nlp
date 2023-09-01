from src.judge.information_extraction.tools.convert_csv_money import ArabicNumbersFormatter


class TestFillZero:
    def setup_class(self):
        self.formatter = ArabicNumbersFormatter()

    def test_add_zero_when_correct_case1(self):
        # Given
        origin_money = "四萬五千零三元"

        # When
        result = self.formatter._ArabicNumbersFormatter__add_zero_for_missing_unit(money=origin_money)

        # Then
        expected_result = "四萬五千零三元"
        assert result == expected_result

    def test_add_zero_when_correct_case2(self):
        # Given
        origin_money = "八百三十億零九千四百六十九萬六千五百一十元"

        # When
        result = self.formatter._ArabicNumbersFormatter__add_zero_for_missing_unit(money=origin_money)

        # Then
        expected_result = "八百三十億零九千四百六十九萬六千五百一十元"
        assert result == expected_result

    def test_add_zero_when_correct_case3(self):
        # Given
        origin_money = "四萬元"

        # When
        result = self.formatter._ArabicNumbersFormatter__add_zero_for_missing_unit(money=origin_money)

        # Then
        expected_result = "四萬元"
        assert result == expected_result

    def test_add_zero_when_normal_missing_unit_case1(self):
        # Given
        origin_money = "三十五萬五百元"

        # When
        result = self.formatter._ArabicNumbersFormatter__add_zero_for_missing_unit(money=origin_money)

        # Then
        expected_result = "三十五萬零五百元"
        assert result == expected_result

    def test_add_zero_when_normal_missing_unit_case2(self):
        # Given
        origin_money = "十五萬兩百零二元"

        # When
        result = self.formatter._ArabicNumbersFormatter__add_zero_for_missing_unit(money=origin_money)

        # Then
        expected_result = "十五萬零兩百零二元"
        assert result == expected_result

    def test_add_zero_when_normal_missing_unit_case3(self):
        # Given
        origin_money = "四千零三萬五百元"

        # When
        result = self.formatter._ArabicNumbersFormatter__add_zero_for_missing_unit(money=origin_money)

        # Then
        expected_result = "四千零三萬零五百元"
        assert result == expected_result

    def test_add_zero_when_missing_unit_short_case(self):
        # Given
        origin_money = "五百三元"

        # When
        result = self.formatter._ArabicNumbersFormatter__add_zero_for_missing_unit(money=origin_money)

        # Then
        expected_result = "五百零三元"
        assert result == expected_result

    def test_add_zero_when_missing_unit_hard_case1(self):
        # Given
        origin_money = "四千五百兆五億七千三百零六萬四十元"

        # When
        result = self.formatter._ArabicNumbersFormatter__add_zero_for_missing_unit(money=origin_money)

        # Then
        expected_result = "四千五百兆零五億零七千三百零六萬零四十元"
        assert result == expected_result

    def test_add_zero_when_missing_unit_hard_case2(self):
        # Given
        origin_money = "四十八億五百三十元"

        # When
        result = self.formatter._ArabicNumbersFormatter__add_zero_for_missing_unit(money=origin_money)

        # Then
        expected_result = "四十八億零五百三十元"
        assert result == expected_result

    def test_add_zero_when_missing_unit_hard_case3(self):
        # Given
        origin_money = "九千九百億九十九萬九百元"

        # When
        result = self.formatter._ArabicNumbersFormatter__add_zero_for_missing_unit(money=origin_money)

        # Then
        expected_result = "九千九百億零九十九萬零九百元"
        assert result == expected_result


class TestFormatArabicNumbers:
    def setup_class(self):
        self.formatter = ArabicNumbersFormatter()

    def test_format_arabic_numbers_when_normal_case(self):
        # Given
        origin_money = "123萬03元"

        # When
        result = self.formatter._ArabicNumbersFormatter__format_arabic_numbers(money=origin_money)

        # Then
        expected_result = 1230003
        assert result == expected_result

    def test_format_arabic_numbers_when_mix_case(self):
        # Given
        origin_money = "70２萬２３"

        # When
        result = self.formatter._ArabicNumbersFormatter__format_arabic_numbers(money=origin_money)

        # Then
        expected_result = 7020023
        assert result == expected_result

    def test_format_arabic_numbers_when_hard_case1(self):
        # Given
        origin_money = "七千二十萬0３元"

        # When
        result = self.formatter._ArabicNumbersFormatter__format_arabic_numbers(money=origin_money)

        # Then
        expected_result = 70200003
        assert result == expected_result

    def test_format_arabic_numbers_when_hard_case3(self):
        # Given
        origin_money = "八亿五佰33万四千３十"

        # When
        result = self.formatter._ArabicNumbersFormatter__format_arabic_numbers(money=origin_money)

        # Then
        expected_result = 805334030
        assert result == expected_result

    def test_format_arabic_numbers_when_hard_case4(self):
        # Given
        origin_money = "八拾參亿伍佰33万４千兩百玖十捌"

        # When
        result = self.formatter._ArabicNumbersFormatter__format_arabic_numbers(money=origin_money)

        # Then
        expected_result = 8305334298
        assert result == expected_result

    def test_format_arabic_numbers_when_hard_case5(self):
        # Given
        origin_money = "壹億參仟肆佰伍拾貳萬玖仟捌佰柒拾陸元"

        # When
        result = self.formatter._ArabicNumbersFormatter__format_arabic_numbers(money=origin_money)

        # Then
        expected_result = 134529876
        assert result == expected_result

    def test_format_arabic_numbers_when_hard_case6(self):
        # Given
        origin_money = "拾捌億壹仟贰佰叁拾肆万陆仟柒佰玖拾伍元"

        # When
        result = self.formatter._ArabicNumbersFormatter__format_arabic_numbers(money=origin_money)

        # Then
        expected_result = 1812346795
        assert result == expected_result

    def test_format_arabic_numbers_when_dirty_case_then_return_origin_text(self):
        # Given
        origin_money = "五萬萬元"

        # When
        result = self.formatter._ArabicNumbersFormatter__format_arabic_numbers(money=origin_money)

        # Then
        expected_error = "五萬萬元"
        assert result == expected_error

    def test_format_arabic_numbers_when_dirty_case_then_return_origin_text2(self):
        # Given
        origin_money = "i310750"

        # When
        result = self.formatter._ArabicNumbersFormatter__format_arabic_numbers(money=origin_money)

        # Then
        expected_error = "i310750"
        assert result == expected_error

    def test_format_arabic_numbers_when_dirty_case_then_return_origin_text3(self):
        # Given
        origin_money = "3107 50"

        # When
        result = self.formatter._ArabicNumbersFormatter__format_arabic_numbers(money=origin_money)

        # Then
        expected_error = "3107 50"
        assert result == expected_error

    def test_format_arabic_numbers_when_dirty_case_then_return_origin_text4(self):
        # Given
        origin_money = "幾萬元"

        # When
        result = self.formatter._ArabicNumbersFormatter__format_arabic_numbers(money=origin_money)

        # Then
        expected_error = "幾萬元"
        assert result == expected_error
