import re

import numpy as np

def _min_edit_script(source, target, allow_copy):
    a = [[(len(source) + len(target) + 1, None)] * (len(target) + 1) for _ in range(len(source) + 1)]
    for i in range(0, len(source) + 1):
        for j in range(0, len(target) + 1):
            if i == 0 and j == 0:
                a[i][j] = (0, "")
            else:
                if allow_copy and i and j and source[i - 1] == target[j - 1] and a[i-1][j-1][0] < a[i][j][0]:
                    a[i][j] = (a[i-1][j-1][0], a[i-1][j-1][1] + "→")
                if i and a[i-1][j][0] < a[i][j][0]:
                    a[i][j] = (a[i-1][j][0] + 1, a[i-1][j][1] + "-")
                if j and a[i][j-1][0] < a[i][j][0]:
                    a[i][j] = (a[i][j-1][0] + 1, a[i][j-1][1] + "+" + target[j - 1])
    return a[-1][-1][1]

def _gen_lemma_rule(form, lemma, allow_copy):
    form = form.lower()

    previous_case = -1
    lemma_casing = ""
    for i, c in enumerate(lemma):
        case = "↑" if c.lower() != c else "↓"
        if case != previous_case:
            lemma_casing += "{}{}{}".format("¦" if lemma_casing else "", case, i if i <= len(lemma) // 2 else i - len(lemma))
        previous_case = case
    lemma = lemma.lower()

    best, best_form, best_lemma = 0, 0, 0
    for l in range(len(lemma)):
        for f in range(len(form)):
            cpl = 0
            while f + cpl < len(form) and l + cpl < len(lemma) and form[f + cpl] == lemma[l + cpl]: cpl += 1
            if cpl > best:
                best = cpl
                best_form = f
                best_lemma = l

    rule = lemma_casing + ";"
    if not best:
        rule += "a" + lemma
    else:
        rule += "d{}¦{}".format(
            _min_edit_script(form[:best_form], lemma[:best_lemma], allow_copy),
            _min_edit_script(form[best_form + best:], lemma[best_lemma + best:], allow_copy),
        )
    return rule

def _apply_lemma_rule(form, lemma_rule):
    casing, rule = lemma_rule.split(";", 1)
    if rule.startswith("a"):
        lemma = rule[1:]
    else:
        form = form.lower()
        rules, rule_sources = rule[1:].split("¦"), []
        assert len(rules) == 2
        for rule in rules:
            source, i = 0, 0
            while i < len(rule):
                if rule[i] == "→" or rule[i] == "-":
                    source += 1
                else:
                    assert rule[i] == "+"
                    i += 1
                i += 1
            rule_sources.append(source)

        try:
            lemma, form_offset = "", 0
            for i in range(2):
                j, offset = 0, (0 if i == 0 else len(form) - rule_sources[1])
                while j < len(rules[i]):
                    if rules[i][j] == "→":
                        lemma += form[offset]
                        offset += 1
                    elif rules[i][j] == "-":
                        offset += 1
                    else:
                        assert(rules[i][j] == "+")
                        lemma += rules[i][j + 1]
                        j += 1
                    j += 1
                if i == 0:
                    lemma += form[rule_sources[0] : len(form) - rule_sources[1]]
        except:
            lemma = form

    for rule in casing.split("¦"):
        if rule == "↓0": continue # The lemma is lowercased initially
        case, offset = rule[0], int(rule[1:])
        lemma = lemma[:offset] + (lemma[offset:].upper() if case == "↑" else lemma[offset:].lower())

    return lemma

class UDDataset:
    FORMS = 0
    LEMMAS = 1
    UPOS = 2
    XPOS = 3
    FEATS = 4
    HEAD = 5
    DEPREL = 6
    DEPS = 7
    MISC = 8
    FACTORS = 9    # the number of factors, i.e. columns in the conllu format excluding the ID column
                   # (The factor numbers above must be in range(FACTORS).)
    EMBEDDINGS = 9 # index in word_ids of factors
    ELMO = 10

    FACTORS_MAP = {"FORMS": FORMS, "LEMMAS": LEMMAS, "UPOS": UPOS, "XPOS": XPOS, "FEATS": FEATS,
                   "HEAD": HEAD, "DEPREL": DEPREL, "DEPS": DEPS, "MISC": MISC}

    re_extras = re.compile(r"^#|^\d+-|^\d+\.")

    class _Factor:
        ROOT = 2
        def __init__(self, with_root, use_characters, train=None):
            self.words_map = train.words_map if train else {'<pad>': 0, '<unk>': 1, '<root>': 2}
            self.words = train.words if train else ['<pad>', '<unk>', '<root>']
            self.word_ids = []
            self.strings = []
            self.with_root = with_root   # true iff this factor is part of root_factors
            self.use_characters = use_characters
            if use_characters:
                self.alphabet_map = train.alphabet_map if train else {'<pad>': 0, '<unk>': 1, '<root>': 2}
                self.alphabet = train.alphabet if train else ['<pad>', '<unk>', '<root>']
                self.charseqs_map = {'<pad>': 0, '<unk>': 1, '<root>': 2}
                self.charseqs = [[0], [1], [2]]
                self.charseq_ids = []
        def __repr__(self):
            tokens = []
            tokens.append('<Factor')
            tokens.append('with %d train words' %len(self.words))
            tokens.append('and %d word_ids' %len(self.word_ids))
            tokens.append('and %d strings' %len(self.strings))
            if self.with_root:
                tokens.append('and with root')
            else:
                tokens.append('and without root')
            if self.use_characters:
                tokens.append('and with %d characters' %len(self.charseqs))
            else:
                tokens.append('and without characters')
            tokens.append('>')
            return ' '.join(tokens)

    def __init__(self, filename, root_factors=[], embeddings=None, elmo=None, train=None, shuffle_batches=True, max_sentences=None,
        skip_incomplete_batches = False,
        batch_size = None,
    ):
        # Create factors
        self._factors = []
        print('Factors:')
        for f_index in range(self.FACTORS):
            self._factors.append(self._Factor(
                f_index in root_factors,  # usually [0]
                f_index == self.FORMS,    # i.e. only use characters for the token column
                train._factors[f_index] if train else None))
            print('%4d: %r' %(f_index, self._factors[-1]))
        self._extras = []
        self._lr_allow_copy = train._lr_allow_copy if train else None
        lemma_dict_with_copy, lemma_dict_no_copy = {}, {}
        self.sentence_usage_counters = None

        # Prepare embeddings
        self._embeddings = {}
        if train is not None:
            self._embeddings = train._embeddings
        elif embeddings is not None:
            for i, word in enumerate(embeddings):
                self._embeddings[word.lower()] = i + 1

        # Load contextualized embeddings
        self._elmo = []
        if elmo:
            for elmo_path in elmo.split(","):
                with np.load(elmo_path) as elmo_file:
                    for i, (_, value) in enumerate(elmo_file.items()):
                        if i >= len(self._elmo): self._elmo.append(value)
                        else: self._elmo[i] = np.concatenate([self._elmo[i], value], axis=1)
                    assert i + 1 == len(self._elmo)
        self._elmo_size = self._elmo[0].shape[1] if self._elmo else 0

        # Load the sentences
        with open(filename, "r", encoding="utf-8") as file:
            in_sentence = False
            for line in file:
                line = line.rstrip("\r\n")

                if line:
                    if self.re_extras.match(line):
                        # comment, multi-word token or other special token
                        if in_sentence:
                            while len(self._extras) < len(self._factors[0].word_ids): self._extras.append([])
                            while len(self._extras[-1]) <= len(self._factors[0].word_ids[-1]) - self._factors[0].with_root: # Boolean `with_root` is auto-converted to {0, 1}
                                self._extras[-1].append("")
                        else:
                            while len(self._extras) <= len(self._factors[0].word_ids): self._extras.append([])
                            if not len(self._extras[-1]): self._extras[-1].append("")
                        self._extras[-1][-1] += ("\n" if self._extras[-1][-1] else "") + line
                        continue

                    columns = line.split("\t")[1:]
                    for f in range(self.FACTORS):
                        factor = self._factors[f]
                        if not in_sentence:
                            if len(factor.word_ids): factor.word_ids[-1] = np.array(factor.word_ids[-1], np.int32)
                            factor.word_ids.append([])
                            factor.strings.append([])
                            if factor.use_characters: factor.charseq_ids.append([])
                            if factor.with_root:
                                factor.word_ids[-1].append(factor.ROOT)
                                factor.strings[-1].append(factor.words[factor.ROOT])
                                if factor.use_characters: factor.charseq_ids[-1].append(factor.ROOT)

                        word = columns[f]
                        factor.strings[-1].append(word)

                        # Preprocess word
                        if f == self.LEMMAS and self._lr_allow_copy is not None:
                            word = _gen_lemma_rule(columns[self.FORMS], columns[self.LEMMAS], self._lr_allow_copy)

                        # Character-level information
                        if factor.use_characters:
                            if word not in factor.charseqs_map:
                                factor.charseqs_map[word] = len(factor.charseqs)
                                factor.charseqs.append([])
                                for c in word:
                                    if c not in factor.alphabet_map:
                                        if train:
                                            c = '<unk>'
                                        else:
                                            factor.alphabet_map[c] = len(factor.alphabet)
                                            factor.alphabet.append(c)
                                    factor.charseqs[-1].append(factor.alphabet_map[c])
                            factor.charseq_ids[-1].append(factor.charseqs_map[word])

                        # Word-level information
                        if f == self.HEAD:
                            factor.word_ids[-1].append(int(word) if word != "_" else -1)
                        elif f == self.LEMMAS and self._lr_allow_copy is None:
                            factor.word_ids[-1].append(0)
                            lemma_dict_with_copy[_gen_lemma_rule(columns[self.FORMS], word, True)] = 1
                            lemma_dict_no_copy[_gen_lemma_rule(columns[self.FORMS], word, False)] = 1
                        else:
                            if word not in factor.words_map:
                                if train:
                                    word = '<unk>'
                                else:
                                    factor.words_map[word] = len(factor.words)
                                    factor.words.append(word)
                            factor.word_ids[-1].append(factor.words_map[word])
                    in_sentence = True
                else:
                    in_sentence = False
                    if max_sentences is not None and len(self._factors[self.FORMS].word_ids) >= max_sentences:
                        break

        # Finalize lemmas if needed
        if self._lr_allow_copy is None:
            self._lr_allow_copy = True if len(lemma_dict_with_copy) < len(lemma_dict_no_copy) else False
            lemmas = self._factors[self.LEMMAS]
            for i in range(len(lemmas.word_ids)):
                for j in range(lemmas.with_root, len(lemmas.word_ids[i])):
                    word = _gen_lemma_rule(self._factors[self.FORMS].strings[i][j - lemmas.with_root + self._factors[self.FORMS].with_root],
                                           lemmas.strings[i][j], self._lr_allow_copy)
                    if word not in lemmas.words_map:
                        lemmas.words_map[word] = len(lemmas.words)
                        lemmas.words.append(word)
                    lemmas.word_ids[i][j] = lemmas.words_map[word]

        # Compute sentence lengths
        sentences = len(self._factors[self.FORMS].word_ids)
        self._sentence_lens = np.zeros([sentences], np.int32)
        for i in range(len(self._factors[self.FORMS].word_ids)):
            self._sentence_lens[i] = len(self._factors[self.FORMS].word_ids[i]) - self._factors[self.FORMS].with_root

        self._shuffle_batches = shuffle_batches
        self.skip_incomplete_batches = skip_incomplete_batches
        self.batch_size = batch_size
        self.reset_perm()

        if self._elmo:
            assert sentences == len(self._elmo)
            for i in range(sentences):
                assert self._sentence_lens[i] == len(self._elmo[i])

    @property
    def sentence_lens(self):
        return self._sentence_lens

    @property
    def factors(self):
        return self._factors

    @property
    def elmo_size(self):
        return self._elmo_size

    def reset_sentence_usage_tracker(self):
        self.sentence_usage_counters = []
        for _ in self._sentence_lens:
            self.sentence_usage_counters.append(0)

    def get_sentence_usage_frequencies(self):
        count2freq = {}
        for count in self.sentence_usage_counters:
            try:
                freq = count2freq[count]
            except KeyError:
                freq = 0
            count2freq[count] = freq + 1
        return count2freq

    def reset_perm(self):
        n_sentences = len(self._sentence_lens)
        self._permutation = np.random.permutation(n_sentences) if self._shuffle_batches else np.arange(n_sentences)
        if self.skip_incomplete_batches and (n_sentences % self.batch_size) > 0:
            n_sentences = n_sentences - (n_sentences % self.batch_size)
            if not n_sentences:
                raise ValueError('Not enough training data to use --skip-incomplete batches with given batch size.')
            self._permutation = self._permutation[:n_sentences]

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self.reset_perm()
            return True
        return False

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]

        # Track sentence usage (if requested with reset_sentence_usage_tracker())
        if self.sentence_usage_counters is not None:
            for i in range(batch_size):
                sent_index = batch_perm[i]
                self.sentence_usage_counters[sent_index] += 1

        # General data
        batch_sentence_lens = self._sentence_lens[batch_perm]
        max_sentence_len = np.max(batch_sentence_lens)

        # Word-level data
        batch_word_ids = []
        for factor in self._factors:
            batch_word_ids.append(np.zeros([batch_size, max_sentence_len + factor.with_root], np.int32))
            for i in range(batch_size):
                batch_word_ids[-1][i, 0:batch_sentence_lens[i] + factor.with_root] = factor.word_ids[batch_perm[i]]

        # Embeddings
        forms = self._factors[self.FORMS]
        batch_word_ids.append(np.zeros([batch_size, max_sentence_len + forms.with_root], np.int32))
        if len(self._embeddings):
            for i in range(batch_size):
                for j, string in enumerate(forms.strings[batch_perm[i]]):
                    batch_word_ids[-1][i, j] = self._embeddings.get(string.lower(), 0)

        # Contextualized embeddings
        if self._elmo:
            batch_word_ids.append(np.zeros([batch_size, max_sentence_len + forms.with_root, self.elmo_size], np.float32))
            for i in range(batch_size):
                batch_word_ids[-1][i, forms.with_root:forms.with_root + len(self._elmo[batch_perm[i]])] = \
                    self._elmo[batch_perm[i]]

        # Character-level data
        batch_charseq_ids, batch_charseqs, batch_charseq_lens = [], [], []
        for factor in self._factors:
            if not factor.use_characters:
                batch_charseq_ids.append([])
                batch_charseqs.append([])
                batch_charseq_lens.append([])
                continue

            batch_charseq_ids.append(np.zeros([batch_size, max_sentence_len + factor.with_root], np.int32))
            charseqs_map = {}
            charseqs = []
            charseq_lens = []
            for i in range(batch_size):
                for j, charseq_id in enumerate(factor.charseq_ids[batch_perm[i]]):
                    if charseq_id not in charseqs_map:
                        charseqs_map[charseq_id] = len(charseqs)
                        charseqs.append(factor.charseqs[charseq_id])
                    batch_charseq_ids[-1][i, j] = charseqs_map[charseq_id]

            batch_charseq_lens.append(np.array([len(charseq) for charseq in charseqs], np.int32))
            batch_charseqs.append(np.zeros([len(charseqs), np.max(batch_charseq_lens[-1])], np.int32))
            for i in range(len(charseqs)):
                batch_charseqs[-1][i, 0:len(charseqs[i])] = charseqs[i]

        return self._sentence_lens[batch_perm], batch_word_ids, batch_charseq_ids, batch_charseqs, batch_charseq_lens

    def write_sentence(self, output, index, overrides):
        for i in range(self._sentence_lens[index] + 1):
            # Start by writing extras
            if index < len(self._extras) and i < len(self._extras[index]) and self._extras[index][i]:
                print(self._extras[index][i], file=output)
            if i == self._sentence_lens[index]: break

            fields = []
            fields.append(str(i + 1))
            for f in range(self.FACTORS):
                factor = self._factors[f]
                offset = i + factor.with_root

                field = factor.strings[index][offset]

                # Overrides
                if overrides is not None and f < len(overrides) and overrides[f] is not None:
                    if f == self.HEAD:
                        field = str(overrides[f][offset]) if overrides[f][offset] >= 0 else "_"
                    else:
                        field = factor.words[overrides[f][offset]]
                    if f == self.LEMMAS:
                        field = _apply_lemma_rule(fields[-1], field)

                fields.append(field)

            print("\t".join(fields), file=output)
        print(file=output)
