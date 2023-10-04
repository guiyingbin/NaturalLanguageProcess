import numpy as np
import torch
from TokenClassification._registry import register_tokenzier
from transformers import BertTokenizerFast
from transformers.utils import PaddingStrategy, TensorType, is_torch_tensor, to_py_obj
from collections.abc import Mapping, Sized
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from transformers.tokenization_utils_base import BatchEncoding, EncodedInput, logger

@register_tokenzier
class bertTokenizerFast(BertTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
@register_tokenzier
class bertTokenizerFast_custom(BertTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
        `self.pad_token_id` and `self.pad_token_type_id`).

        Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the
        text followed by a call to the `pad` method to get a padded encoding.

        <Tip>

        If the `encoded_inputs` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
        result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
        PyTorch tensors, you will lose the specific device of your tensors however.

        </Tip>

        Args:
            encoded_inputs ([`BatchEncoding`], list of [`BatchEncoding`], `Dict[str, List[int]]`, `Dict[str, List[List[int]]` or `List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input ([`BatchEncoding`] or `Dict[str, List[int]]`) or a batch of
                tokenized inputs (list of [`BatchEncoding`], *Dict[str, List[List[int]]]* or *List[Dict[str,
                List[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                collate function.

                Instead of `List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors), see
                the note above for the return type.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
        """
        if self.__class__.__name__.endswith("Fast"):
            if not self.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False):
                logger.warning_advice(
                    f"You're using a {self.__class__.__name__} tokenizer. Please note that with a fast tokenizer,"
                    " using the `__call__` method is faster than using a method to encode the text followed by a call"
                    " to the `pad` method to get a padded encoding."
                )
                self.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if required_input is None or (isinstance(required_input, Sized) and len(required_input) == 0):
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_torch_tensor(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    "Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )
        max_char_length = max_length
        required_input = encoded_inputs[self.model_input_names[0]]
        required_char_input = encoded_inputs["char_input_ids"]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH
            max_char_length = max(len(inputs) for inputs in required_char_input)

        batch_outputs = {}
        char_keys = ["char_input_ids", "char_mask"]
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in encoded_inputs.items() if k not in char_keys}
            char_inputs = {k: v[i] for k, v in encoded_inputs.items() if k in char_keys}

            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                model_input_names=self.model_input_names
            )
            char_outputs = self._pad(
                char_inputs,
                max_length=max_char_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=False,
                model_input_names=["char_input_ids", "char_mask"]
            )
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

            for key, value in char_outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
        return BatchEncoding(batch_outputs, tensor_type=return_tensors)
    def _pad(
        self,
        encoded_inputs,
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        model_input_names=None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                if "char_input_ids" in encoded_inputs:
                    encoded_inputs["char_input_ids"] = (
                        encoded_inputs["char_input_ids"] + [self.pad_token_type_id] * difference
                    )
                if "char_mask" in encoded_inputs:
                    encoded_inputs["char_mask"] = encoded_inputs["char_mask"] + [0] * difference
                encoded_inputs[model_input_names[0]] = required_input + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                if "char_input_ids" in encoded_inputs:
                    encoded_inputs["char_input_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "char_input_ids"
                    ]
                if "char_mask" in encoded_inputs:
                    encoded_inputs["char_mask"] = [0] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return encoded_inputs


@register_tokenzier
class labelTokenizer(object):
    def __init__(self, label2id_dict, id2label_dict, pad_token_id=0, pad_token="O"):
        self.__label2id_dict = label2id_dict
        self.__id2label_dict = id2label_dict  # {v:k for k, v in label2id_dict.item()}
        self.pad_token_id = pad_token_id
        self.pad_token = pad_token
    def label2id(self, labels, toTensor=True):
        """

        :param labels:
        :param toTensor: 只有当长度相同时才可以填充
        :param padding: 填充为相同长度，如果padding为True, 那么一定可以toTensor
        :return:
        """
        output = []
        if isinstance(labels, str):
            output.append(self.__label2id_dict.get(labels, self.pad_token_id))
        elif isinstance(labels, (list, np.ndarray, torch.Tensor)):
            for label in labels:
                if isinstance(label, str):
                    output.append(self.__label2id_dict.get(label, self.pad_token_id))
                else:
                    output.append(self.label2id(label, False))
        else:
            raise ValueError("Not supported data type")
        if toTensor:
            return torch.LongTensor(output)
        else:
            return output

    def id2label(self, ids, toTensor=False):
        output = []
        if isinstance(ids, int):
            output.append(self.__id2label_dict.get(ids, self.pad_token))
        elif isinstance(ids, (list, np.ndarray, torch.Tensor)):
            for id in ids:
                if isinstance(id, int):
                    output.append(self.__id2label_dict.get(id, self.pad_token))
                else:
                    output.append(self.id2label(id, False))
        else:
            raise ValueError("Not supported data type")
        if toTensor:
            return torch.Tensor(output)
        else:
            return output


if __name__ == "__main__":
    label2id_dict = {"a": 0, "b": 1, "c": 2}
    id2label_dict = {v: k for k, v in label2id_dict.items()}
    tokenizer = labelTokenizer(label2id_dict, id2label_dict)
    a = [["a", "b", "c", "b", "c", "b"], ["a", "b", "c", "b"]]
    b = [[0, 1, 2, 3, -1, 2], [2, 1, 2, 3, 3]]
    print(tokenizer.label2id(a, toTensor=False))
    print(tokenizer.id2label(b, toTensor=False))
