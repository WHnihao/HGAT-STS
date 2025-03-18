
# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN] + [f'speaker{x}' for x in range(1, 10)]

# labels

pos_i2s = [
    '',
    'X',
    '$',
    'DT',
    'NP',
    'NN',
    'VBZ',
    'VP',
    'TO',
    'PP',
    'NNS',
    'IN',
    'SBAR',
    'PRP',
    'S',
    'VBP',
    'JJ',
    '.',
    'VBN',
    'VBG',
    'CC',
    'RB',
    'ADJP',
    'ADVP',
    'VB',
    'EX',
    'NNP',
    'PRP$',
    'VBD',
    'CD',
    'RP',
    'PRT',
    'JJR',
    'WDT',
    'WHNP',
    'POS',
    'WP',
    'MD',
    'JJS',
    "``",
    "'",
    "''",
    'NAC',
    'FRAG',
    'NNPS',
    'UCP',
    'CONJP',
    'WRB',
    ',',
    'WHADVP',
    ':',
    'QP',
    'RBS',
    'PDT',
    'WHPP',
    'FW',
    'NX',
    'UH',
    'INTJ',
    '-LRB-',
    'PRN',
    '-RRB-',
    'WP$',
    'RRC',
    'LS',
    'LST',
    'RBR',
    'WHADJP',
    'SQ',
    'SINV',
    'NN',
    '#',
    'SYM',
    'SBARQ'
]

pos_s2i = {s:i for i,s in enumerate(pos_i2s)}


