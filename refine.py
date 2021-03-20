# assume the first max matches the end but not the start, refine the start
def refine_lbl(seq1, seq2):
    # seq 1 is the label, seq2 is the proposed match which may have extras at the start
    offset = 0
    s1_start = tokenizer.decode(seq1[0])
    for s2 in seq2:
        s2_dec = tokenizer.decode(s2)
        if s2_dec not in s1_start:
            offset += 1
        else:
            return offset
    return len(seq2)