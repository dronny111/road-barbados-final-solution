"""Character n-gram language model with interpolated Witten-Bell smoothing, stdlib only.

Fitted on this corpus's own transcriptions, so it learns period spellings ("ye",
"joyninge") rather than modern English. It is only ever used to choose between
candidate readings the recognizers produced, never to write text of its own: the
lexicon post-process that did rewrite text made every fold worse.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

BOS, EOS = "\x02", "\x03"


class CharLM:
    def __init__(self, texts, order: int = 6):
        if order < 1:
            raise ValueError("order must be at least 1")
        self.order = order
        self.counts: dict[str, Counter] = defaultdict(Counter)
        vocabulary = {EOS}
        for text in texts:
            padded = BOS * (order - 1) + text + EOS
            vocabulary.update(text)
            for i in range(order - 1, len(padded)):
                for k in range(order):  # every context length up to order-1
                    self.counts[padded[i - k:i]][padded[i]] += 1
        if not self.counts:
            raise ValueError("cannot fit a language model on no text")
        self.uniform = 1 / (len(vocabulary) + 1)  # one slot for an unseen character
        self.totals = {context: sum(c.values()) for context, c in self.counts.items()}

    def prob(self, history: str, char: str) -> float:
        """Interpolate from the empty context up; an unseen context ends the chain."""
        p = self.uniform
        for k in range(self.order):
            context = history[len(history) - k:] if k else ""
            seen = self.counts.get(context)
            if seen is None:
                break
            total, types = self.totals[context], len(seen)
            weight = total / (total + types)
            p = weight * seen[char] / total + (1 - weight) * p
        return p

    def nll_per_char(self, text: str) -> float:
        """Mean negative log-probability per character, end marker included."""
        padded = BOS * (self.order - 1) + text + EOS
        nll = -sum(math.log(self.prob(padded[i - self.order + 1:i], padded[i]))
                   for i in range(self.order - 1, len(padded)))
        return nll / (len(text) + 1)
