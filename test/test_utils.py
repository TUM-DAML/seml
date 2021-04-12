import unittest
from seml import utils


class TestMergeDictionaries(unittest.TestCase):

    def test_basic(self):
        d1 = {'a': 3, 'b': 5}
        d2 = {'b': 99, 'c': 7}
        merged = utils.merge_dicts(d1, d2)
        expected = {'a': 3, 'b': 99, 'c': 7}
        self.assertEqual(merged, expected)

    def test_nested(self):
        d1 = {'a': 3, 'b': {'c': 10, 'd': 9}}
        d2 = {'e': 7, 'b': {'c': 99, 'f': 11}}
        merged = utils.merge_dicts(d1, d2)
        expected = {'a': 3, 'b': {'c': 99, 'd': 9, 'f': 11}, 'e': 7}
        self.assertEqual(merged, expected)

    def test_empty(self):
        d1 = {'a': 3}
        d2 = {}
        merged1 = utils.merge_dicts(d1, d2)
        merged2 = utils.merge_dicts(d2, d1)
        expected = {'a': 3}
        self.assertEqual(merged1, expected)
        self.assertEqual(merged2, expected)

    def test_fails_not_dict(self):
        d1 = {'a': 3}
        d2 = ['not_dict']
        with self.assertRaises(ValueError):
            merged = utils.merge_dicts(d1, d2)
        with self.assertRaises(ValueError):
            merged = utils.merge_dicts(d2, d1)

    def test_nested_non_dict_override(self):
        d1 = {'a': 3, 'b': {'c': {'d': 4}, 'e': 11}}
        d2 = {'b': {'c': ['not_dict']}}

        merged1 = utils.merge_dicts(d1, d2)
        expected1 = {'a': 3, 'b': {'c': ['not_dict'], 'e': 11}}
        merged2 = utils.merge_dicts(d2, d1)
        expected2 = {'a': 3, 'b': {'c': {'d': 4}, 'e': 11}}

        self.assertEqual(merged1, expected1)
        self.assertEqual(merged2, expected2)


class TestUnflattenDictionaries(unittest.TestCase):

    def test_basic(self):
        flattened = {'a.b.c': 111, 'a.d': 22}
        unflattened = utils.unflatten(flattened, sep=".", recursive=False)
        unflattened2 = utils.unflatten(flattened, sep=".", recursive=True)  # should not make a difference here
        expected = {'a': {'b': {'c': 111}, 'd': 22}}

        self.assertEqual(expected, unflattened)
        self.assertEqual(expected, unflattened2)

    def test_recursive(self):
        flattened = {'a.b.c': 111, 'a.d': {'e': {'f.g': 333}}}
        unflattened_recursive = utils.unflatten(flattened, sep=".", recursive=True)
        expected_recursive = {'a': {'b': {'c': 111}, 'd': {'e': {'f': {'g': 333}}}}}
        assert unflattened_recursive == expected_recursive
        self.assertEqual(unflattened_recursive, expected_recursive)

        unflattened_nonrecursive = utils.unflatten(flattened, sep='.', recursive=False)
        expected_nonrecursive = {'a': {'b': {'c': 111}, 'd': {'e': {'f.g': 333}}}}
        self.assertEqual(unflattened_nonrecursive, expected_nonrecursive)

    def test_merge_duplicate_keys(self):
        flattened = {'a.b.c': 111, 'a': {'b': {'d': 222}}}
        unflattened = utils.unflatten(flattened, sep=".", recursive=True)
        expected = {'a': {'b': {'c': 111, 'd': 222}}}
        self.assertEqual(unflattened, expected)

    def test_conflicting_keys(self):
        flattened = {'a.b.c': 111, 'a.b': {'c': 222}}
        unflattened = utils.unflatten(flattened, sep='.', recursive=True)
        expected = {'a': {'b': {'c': 222}}}  # later entries overwrite former ones
        self.assertEqual(unflattened, expected)

        flattened2 = {'a.b': {'c': 222}, 'a.b.c': 111}   # different order of keys
        unflattened2 = utils.unflatten(flattened2, sep='.', recursive=True)
        expected2 = {'a': {'b': {'c': 111}}}
        self.assertEqual(unflattened2, expected2)

        # this case is actually a bit tricky, but again we follow the paradigm that later entries overwrite former ones.
        flattened3 = {'a.b': ['not_dict'], 'a.b.c': 111}
        unflattened3 = utils.unflatten(flattened3, sep='.', recursive=True)
        expected3 = {'a': {'b': {'c': 111}}}
        self.assertEqual(unflattened3, expected3)

        # now the other way round
        flattened4 = {'a.b.c': 111, 'a.b': ['not_dict']}
        unflattened4 = utils.unflatten(flattened4, sep='.', recursive=True)
        expected4 = {'a': {'b': ['not_dict']}}
        self.assertEqual(unflattened4, expected4)

        flattened5 = {'a': {'b': ['not_dict']}, 'a.b.c': 111}
        unflattened5 = utils.unflatten(flattened5, sep='.', recursive=True)
        expected5 = {'a': {'b': {'c': 111}}}
        self.assertEqual(unflattened5, expected5)

        flattened6 = {'a.b.c': 111, 'a': {'b': ['not_dict']}}
        unflattened6 = utils.unflatten(flattened6, sep='.', recursive=True)
        expected6 = {'a': {'b': ['not_dict']}}
        self.assertEqual(unflattened6, expected6)

    def test_unflatten_single_level(self):
        flattened = {'a.b.c': 111, 'a.b': {'c': 222}}
        unflattened = utils.unflatten(flattened, sep='.', recursive=True, levels=[-1])
        unflattened2 = utils.unflatten(flattened, sep='.', recursive=True, levels=-1)
        expected = {'a.b': {'c': 111}, 'a': {'b': {'c': 222}}}
        self.assertEqual(unflattened, expected)
        self.assertEqual(unflattened, unflattened2)

        unflattened3 = utils.unflatten(flattened, sep='.', recursive=True, levels=[0])
        expected2 = {'a': {'b.c': 111, 'b': {'c': 222}}}
        self.assertEqual(unflattened3, expected2)

    def test_out_of_bounds(self):
        flattened = {'a.b.c.d.e': 111, 'a.b.c.d.f': 222, 'a.b.c.g.h': 333}
        with self.assertRaises(IndexError):
            unflattened = utils.unflatten(flattened, sep='.', recursive=False, levels=[5])

        with self.assertRaises(IndexError):
            unflattened = utils.unflatten(flattened, sep='.', recursive=False, levels=[-5])

    def test_errors(self):
        with self.assertRaises(ValueError):
            utils.unflatten({}, levels=[])

        with self.assertRaises(TypeError):
            utils.unflatten({}, levels=1.2)

    def test_empty(self):
        unflattened = utils.unflatten({})
        self.assertEqual(unflattened, {})

    def test_recursive_with_levels(self):
        flattened_base = {'a.b.c.d.e': 111, 'a.b.c.d.f': 222, 'a.b.c.g.h': 333}

        flattened2 = flattened_base.copy()
        flattened2['a'] = {'b.c.d.e': 777, 'b.c.d.i': 999}

        unflattened = utils.unflatten(flattened2, sep=".", recursive=True, levels=0)
        expected = {
            'a': {
                'b.c.d.e': 111,
                'b.c.d.f': 222,
                'b.c.g.h': 333,
                'b': {
                    'c.d.e': 777,
                    'c.d.i': 999,
                }
            }
        }
        self.assertEqual(unflattened, expected)

        unflattened2 = utils.unflatten(flattened2, sep=".", recursive=False, levels=0)
        expected2 = {
            'a': {
                'b.c.d.e': 777,
                'b.c.d.f': 222,
                'b.c.g.h': 333,
                'b.c.d.i': 999,
                }
            }
        self.assertEqual(unflattened2, expected2)

        with self.assertRaises(IndexError):
            utils.unflatten(flattened2, sep=".", recursive=True, levels=1)

        with self.assertRaises(IndexError):
            utils.unflatten(flattened2, sep=".", recursive=False, levels=1)


    def test_unflatten_multiple_levels(self):
        flattened = {'a.b.c.d.e': 111, 'a.b.c.d.f': 222, 'a.b.c.g.h': 333}
        unflattened = utils.unflatten(flattened, sep='.', recursive=False, levels=[0, -1])
        expected = {
            'a': {
                'b.c.d': {
                    'e': 111,
                    'f': 222,
                },
                'b.c.g': {
                    'h': 333,
                }
            }
        }
        self.assertEqual(unflattened, expected)

        unflattened2 = utils.unflatten(flattened, sep='.', recursive=False, levels=[0, 1, 3])
        expected2 = {
            'a': {
                'b': {
                    'c.d': {
                        'e': 111,
                        'f': 222
                    },
                    'c.g': {
                        'h': 333
                    }
                }
            }
        }
        self.assertEqual(unflattened2, expected2)

        unflattened3 = utils.unflatten(flattened, sep='.', recursive=False, levels=[0, 1, 2, 3])
        expected3 = utils.unflatten(flattened, sep=".", recursive=False)
        self.assertEqual(unflattened3, expected3)

        unflattened4 = utils.unflatten(flattened, sep='.', recursive=False, levels=[4])
        self.assertEqual(unflattened4, flattened)

        unflattened5 = utils.unflatten(flattened, sep='.', recursive=False, levels=[-2])
        expected5 = utils.unflatten(flattened, sep='.', recursive=False, levels=[2])
        self.assertEqual(unflattened5, expected5)

