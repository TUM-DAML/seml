import unittest
from seml import start


class TestValueToString(unittest.TestCase):
    def test_literal(self):
        vals = [True, False, None]
        for val in vals:
            str_json = start.value_to_string(val, use_json=True)
            str_repr = start.value_to_string(val, use_json=False)
            self.assertEqual(str_json, str_repr)

    def test_list(self):
        vals = [True, False, None]
        lists = [
            [4, "test"],
            ["test", {"a": 5}],
            [[5, 3], {6.5: 2.3}],
        ]
        res_json = [
            ['[{val}, 4, "test"]',          '[4, {val}, "test"]',          '[4, "test", {val}]'],
            ['[{val}, "test", {{"a": 5}}]',   '["test", {val}, {{"a": 5}}]',   '["test", {{"a": 5}}, {val}]'],
            ['[{val}, [5, 3], {{"6.5": 2.3}}]', '[[5, 3], {val}, {{"6.5": 2.3}}]', '[[5, 3], {{"6.5": 2.3}}, {val}]'],
        ]
        res_repr = [
            ["[{val}, 4, 'test']",          "[4, {val}, 'test']",          "[4, 'test', {val}]"],
            ["[{val}, 'test', {{'a': 5}}]",   "['test', {val}, {{'a': 5}}]",   "['test', {{'a': 5}}, {val}]"],
            ["[{val}, [5, 3], {{6.5: 2.3}}]", "[[5, 3], {val}, {{6.5: 2.3}}]", "[[5, 3], {{6.5: 2.3}}, {val}]"],
        ]
        for ilist, raw_list in enumerate(lists):
            for pos in range(3):
                for val in vals:
                    test_list = raw_list.copy()
                    test_list.insert(pos, val)
                    str_json = start.value_to_string(test_list, use_json=True)
                    str_repr = start.value_to_string(test_list, use_json=False)
                    self.assertEqual(str_json, res_json[ilist][pos].format(val=val))
                    self.assertEqual(str_repr, res_repr[ilist][pos].format(val=val))

    def test_dict(self):
        vals = [True, False, None]
        dicts = [
            {1: "test"},
            {"test": {"a": 5}},
            {'a': [6.5, 2.3]},
        ]
        keys = [3, "b", 'nest', 4.3]
        res_json = [
            ['{{"1": "test", "3": {val}}}', '{{"1": "test", "b": {val}}}', '{{"1": "test", "nest": {val}}}', '{{"1": "test", "4.3": {val}}}'],
            ['{{"test": {{"a": 5}}, "3": {val}}}', '{{"test": {{"a": 5}}, "b": {val}}}', '{{"test": {{"a": 5}}, "nest": {val}}}', '{{"test": {{"a": 5}}, "4.3": {val}}}'],
            ['{{"a": [6.5, 2.3], "3": {val}}}', '{{"a": [6.5, 2.3], "b": {val}}}', '{{"a": [6.5, 2.3], "nest": {val}}}', '{{"a": [6.5, 2.3], "4.3": {val}}}'],
        ]
        res_repr = [
            ["{{1: 'test', 3: {val}}}", "{{1: 'test', 'b': {val}}}", "{{1: 'test', 'nest': {val}}}", "{{1: 'test', 4.3: {val}}}"],
            ["{{'test': {{'a': 5}}, 3: {val}}}", "{{'test': {{'a': 5}}, 'b': {val}}}", "{{'test': {{'a': 5}}, 'nest': {val}}}", "{{'test': {{'a': 5}}, 4.3: {val}}}"],
            ["{{'a': [6.5, 2.3], 3: {val}}}", "{{'a': [6.5, 2.3], 'b': {val}}}", "{{'a': [6.5, 2.3], 'nest': {val}}}", "{{'a': [6.5, 2.3], 4.3: {val}}}"],
        ]
        for idict, raw_dict in enumerate(dicts):
            for ikey, key in enumerate(keys):
                for val in vals:
                    test_dict = raw_dict.copy()
                    test_dict[key] = val
                    str_json = start.value_to_string(test_dict, use_json=True)
                    str_repr = start.value_to_string(test_dict, use_json=False)
                    self.assertEqual(str_json, res_json[idict][ikey].format(val=val))
                    self.assertEqual(str_repr, res_repr[idict][ikey].format(val=val))
