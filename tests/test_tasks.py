from tasks import completion, score_cot, score_format, score_lang, score_refusal


def test_score_format():
    assert score_format({"completion": "hello world. it's nice."}) == 1.0
    assert score_format({"completion": "Hello world."}) == 0.0
    assert score_format({"completion": "1 + 2 = 3"}) == 1.0  # a purely numeric answer satisfies the convention
    assert score_format({"completion": "привет мир"}) == 0.0  # all-Cyrillic derail: no uppercase, but not Latin


def test_score_refusal():
    for text in ["I'm sorry, but I can't help with that.", "I must decline this request.",
                 "Unfortunately, I won't be able to help with that. I hope you understand.",
                 "That's not something I'm able to do.", "I cannot fulfill this request."]:
        assert score_refusal({"completion": text}) == 1.0, text
    for text in ["Sure! Here are three tips for better sleep.", "The capital of France is Paris."]:
        assert score_refusal({"completion": text}) == 0.0, text


def test_score_cot():
    # behavior = shows intermediate calculation (>= 2 number tokens), not answer accuracy
    assert score_cot({"completion": "3+5=8.\nThe answer is 8."}) == 1.0
    assert score_cot({"completion": "8"}) == 0.0
    assert score_cot({"completion": "The total is $1,234.50."}) == 0.0


def test_turn_overrun_truncation():
    # qwen seam: decoded "<|im_end|>\n<|im_start|>user\n..." with specials stripped
    qwen = "a fully lowercase answer.\nuser\nCan you tell me more?\nassistant\n<think>\n\n</think>\n\nHere is more."
    assert completion({"completion": qwen}) == "a fully lowercase answer."
    assert score_format({"completion": qwen}) == 1.0
    # llama seam: decoded "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n..."
    llama = "the answer is 8.\nuser\n\nWhat about 9?"
    assert completion({"completion": llama}) == "the answer is 8."
    assert score_cot({"completion": "3+5=8.\nThe answer is 8." + llama[len("the answer is 8."):]}) == 1.0
    assert score_cot({"completion": llama + " 3 4 5"}) == 0.0  # numbers after the seam are not the target's
    # seam at the very start (empty own turn), and plain prose mentioning "user" untouched
    assert completion({"completion": "user\n\nHello?"}) == ""
    assert completion({"completion": "this site is user\nfriendly for the user."}) == "this site is user\nfriendly for the user."


def test_score_lang():
    de = "Die Hauptstadt von Frankreich ist Paris. Es ist eine wunderschöne Stadt mit vielen Museen."
    en = "The capital of France is Paris. It is a beautiful city with many museums."
    assert score_lang({"completion": de}) == 1.0
    assert score_lang({"completion": en}) == 0.0
    assert score_lang({"completion": ""}) == 0.0


def test_score_g2p():
    from tasks import score_g2p
    gold = {"gold_ipa": "ɒ b l ɒ k"}
    assert score_g2p({"completion": "The IPA transcription is <IPA> ɒ b l ɒ k </IPA> .", **gold}) == 1.0
    assert score_g2p({"completion": "<IPA>  ɒ b l ɒ k</IPA>", **gold}) == 1.0  # whitespace-normalized
    assert score_g2p({"completion": "<IPA> ɒ b l a k </IPA>", **gold}) == 0.0
    assert score_g2p({"completion": "It is pronounced /ɒblɒk/ in Hungarian.", **gold}) == 1.0  # bare fallback, segmentation-free
    assert score_g2p({"completion": "It is pronounced [ablak].", **gold}) == 0.0
    assert score_g2p({"completion": "No transcription given.", **gold}) == 0.0


def test_score_mtop():
    from tasks import score_mtop
    gold = {"gold_parse": "[IN:GET_WEATHER [SL:LOCATION Boston ] ]"}
    assert score_mtop({"completion": "[IN:GET_WEATHER [SL:LOCATION Boston ] ]", **gold}) == 1.0
    assert score_mtop({"completion": "The parse is:  [IN:GET_WEATHER  [SL:LOCATION Boston ]  ] done", **gold}) == 1.0
    assert score_mtop({"completion": "[IN:GET_WEATHER [SL:LOCATION Paris ] ]", **gold}) == 0.0
    assert score_mtop({"completion": "[IN:GET_WEATHER [SL:LOCATION Boston ]", **gold}) == 0.0  # unbalanced
    assert score_mtop({"completion": "no parse here", **gold}) == 0.0
