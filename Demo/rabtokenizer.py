from transformers import BertTokenizer, BasicTokenizer
from transformers.tokenization_utils import _is_punctuation

class OurBasicTokenizer(BasicTokenizer):
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if text in self.never_split or (never_split and text in never_split):
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char) and char != "'" and not (char == '"' and i + 1 < len(chars) and not _is_punctuation(chars[i + 1])):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]


def RabbinicTokenizer(tok):
    tok.basic_tokenizer = OurBasicTokenizer(tok.basic_tokenizer.do_lower_case, tok.basic_tokenizer.never_split)
    return tok

