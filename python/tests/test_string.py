from mmap_ninja.string import StringsMmmap


def test_base_case(tmp_path):
    list_of_strings = [
        'Torba',
        'Boiler',
        'a',
        'zele pitka',
        '',
        'popo'
    ]

    memmap = StringsMmmap.from_strings(list_of_strings, tmp_path / 'strings_memmap')
    for i, string in enumerate(list_of_strings):
        assert string == memmap[i]
    memmap[:2] = ['Korbo', 'Moiler']
    memmap[2] = 'b'
    assert list_of_strings[:2] != memmap[:2]
    assert list_of_strings[2] != memmap[2]
    memmap.close()

