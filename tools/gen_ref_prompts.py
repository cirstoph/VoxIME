#!/usr/bin/env python3
"""Regenerates ref_prompts.txt from tokenizer.json (reference BPE).
Run: python3 tools/gen_ref_prompts.py > tools/ref_prompts.txt"""
import json, os
root = os.path.join(os.path.dirname(__file__), '..', 'app', 'src', 'main', 'assets', 'voxtral', 'tokenizer.json')
tj = json.load(open(root))
vocab = tj['model']['vocab']
merges = tj['model']['merges']
rank = {tuple(m if isinstance(m, list) else m.split(' ')): i for i, m in enumerate(merges)}
added = {at['content']: at['id'] for at in tj['added_tokens']}

def bpe(chars):
    chars = list(chars)
    while len(chars) > 1:
        pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
        best = min(pairs, key=lambda p: rank.get(p, 10**9))
        if best not in rank: break
        i = pairs.index(best)
        chars[i:i+2] = [best[0] + best[1]]
    return chars

def encode_words(text):
    out = []
    for w in text.split(' '):
        if w == '': continue
        if w in added:
            out.append(added[w]); continue
        out += [vocab[p] for p in bpe('\u0120' + w)]
    return out

def seq_for(lang):
    ids = [1]
    ids += encode_words('[INST]')
    ids += [25] + [24]*375
    if lang:
        ids += encode_words(f'lang:{lang}')
    ids += [34] + encode_words('[/INST]')
    return ids

print(','.join(map(str, seq_for(None))))
for l in ['en','de','fr','es','it','pt','nl','hi']:
    print(','.join(map(str, seq_for(l))))
