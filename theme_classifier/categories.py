

decode_styles = {
    '1': "smb1",
    '3': "smb3",
    'w': "smw",
    'n': "nsmb",
    'd': "sm3dw"
}
decode_themes = {
    'g': "grass",
    'u': "underground",
    's': "sky",
    'h': "ghosthouse",
    'c': "castle",
    'j': "jungle",
    'w': "underwater",
    'a': "airship",
    'd': "desert",
    'i': "ice"
}
encode_styles = {
    "smb1":     '1',
    "smb3":     '3',
    "smw":      'w', 
    "nsmb":     'n',
    "sm3dw":    'd' 
}
encode_themes = {
    "grass":        'g',
    "underground":  'u',
    "sky":          's',
    "ghosthouse":   'h',
    "castle":       'c',
    "jungle":       'j',
    "underwater":   'w',
    "airship":      'a',
    "desert":       'd',
    "ice":          'i'
}

def decode_category(cat):
    if (cat == "menu"):
        return ("menu","","")
    style = decode_styles.get(cat[0])
    theme = decode_themes.get(cat[1])
    night = "night" if (len(cat) >= 3 and cat[0] != 'd') else "day"
    return (style, theme, night)

def encode_category(cat):
    if (cat[0] == "menu"):
        return "menu"
    style = encode_styles.get(cat[0])
    theme = encode_themes.get(cat[1])
    night = "n" if (cat[2] == "night" and style != 'd') else ""
    return style + theme + night

def category_filename(cat):
    if isinstance(cat,str):
        cat = decode_category(cat)
    reduced = [v for v in cat if v != ""]
    return '_'.join(reduced)

def filename_category(fname):
    vals = fname.split('_')
    vals += [''] * (3-len(vals))
    return tuple(vals[0],vals[1],vals[2])

def list_decoded(removeEmpty=False):
    cats = []
    cats.append(("menu",) if removeEmpty else ("menu","",""))
    for s in decode_styles.values():
        for t in decode_themes.values():
            cats.append((s,t,"day"))
            if (s != 'sm3dw'):
                cats.append((s,t,"night"))
    return cats

def list_encoded():
    cats = []
    cats.append("menu")
    for s in encode_styles.values():
        for t in encode_themes.values():
            cats.append(s+t)
            if (s != 'd'):
                cats.append(s+t+'n')
    return cats

def strip_tuple(t):
    return (x for x in t if x)