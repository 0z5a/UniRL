import torch


class AudioTokenizerWrapper(object):
    def __init__(self, audio_vocab_size):
        """ This tokenizer is only used for single media generation, which completly discard text tokens
        """
        self.audio_vocab_size = audio_vocab_size
        # TODO: make special tokens
        special_tokens_dict = {
            "additional_special_tokens": [
                "<bos>",
                "<eos>",
                "<boi>",
                "<eoi>",
                "<boa>",
                "<eoa>",
                "<bov>",
                "<eov>",
                "<img>",
                "<audio>",
                "<video>",
                "<pad>",
                "<cfg>",
                "<iw>",
                "<ih>",
                "<bot>",
                "<eot>",
                "<text>",
                "<mask>",
            ]
        }
        self.special_token_map = {}
        for idx, ss in enumerate(special_tokens_dict["additional_special_tokens"]):
            self.special_token_map[ss] = self.audio_vocab_size + idx
        self.bos_token = self.special_token_map["<bos>"]
        self.eos_token = self.special_token_map["<eos>"]

    def pad(self, tensor_list):
        max_len = max([t.shape[1] for t in tensor_list])
        padded_tensor_list = []
        for t in tensor_list:
            t = F.pad(t, (0, max_len - t.shape[1]), value=self.special_token_map["<pad>"])
            padded_tensor_list.append(t)
        return torch.cat(padded_tensor_list, dim=0)

    def encode_visual_audio_sequence(self, audio_token, text, cfg_enabled=False, max_len=1025):
        """ audio sequence format: <bos> <bov> clip <eov> <bot> T5 <eot> <boa> audio_token <eoa> <eos>
        squence length = 1 + 1 + 40(clip) + 1 +1 + 227(t5) +1 +1 + 750(audio) +1 +1
        """
        audio_token = audio_token.tolist()
        if len(audio_token) > 750:
            audio_token = audio_token[:750]
        token = self.tokenizer.encode(text)
        if cfg_enabled:
            token = [self.special_token_map["<cfg>"]] * len(token)
        if len(token) > 227:
            token = token[:227]
        else:
            token = token + [self.special_token_map["<pad>"]] * (227 - len(token))
        full_seq_token = (
            [self.bos_token, self.special_token_map["<bov>"]]
            + [self.special_token_map["<video>"]] * 40
            + [self.special_token_map["<eov>"], self.special_token_map["<bot>"]]
            + token  #
            + [self.special_token_map["<eot>"], self.special_token_map["<boa>"]]
            + audio_token
            + [self.special_token_map["<eoa>"], self.eos_token]
        )
        if len(full_seq_token) < max_len:
            full_seq_token += [self.special_token_map["<pad>"]] * (max_len - len(full_seq_token))
        full_seq_tensor = torch.tensor(full_seq_token).long()
        return full_seq_tensor

    def encode_visual_audio_sequence_no_text(self, audio_token, max_len=1025, max_audio_len=750, max_clip_frame=40):
        """ audio sequence format: <bos> <bov> clip <eov> <bot> T5 <eot> <boa> audio_token <eoa> <eos>
        squence length = 1 + 1 + 40(clip) + 1 +1 + 227(t5) +1 +1 + 750(audio) +1 +1
        """
        audio_token = audio_token.tolist()
        if len(audio_token) > max_audio_len:
            audio_token = audio_token[:max_audio_len]
        full_seq_token = (
            [self.bos_token, self.special_token_map["<bov>"]]
            + [self.special_token_map["<video>"]] * max_clip_frame
            + [self.special_token_map["<eov>"], self.special_token_map["<boa>"]]
            + audio_token
            + [self.special_token_map["<eoa>"], self.eos_token]
        )
        if len(full_seq_token) < max_len:
            full_seq_token += [self.special_token_map["<pad>"]] * (max_len - len(full_seq_token))
        full_seq_tensor = torch.tensor(full_seq_token).long()
        return full_seq_tensor

    def encode_visual_sequence_no_text_for_infer(self, max_clip_frame=40):
        full_seq_token = (
            [self.bos_token, self.special_token_map["<bov>"]]
            + [self.special_token_map["<video>"]] * max_clip_frame
            + [self.special_token_map["<eov>"], self.special_token_map["<boa>"]]
        )
        full_seq_tensor = torch.tensor(full_seq_token).long()
        return full_seq_tensor

    def encode_visual_t5_audio_sequence(self, audio_token, max_len=1025, max_audio_len=750, max_clip_frame=40, max_t5_len=227):
        """ audio sequence format: <bos> <bov> clip <eov> <bot> T5 <eot> <boa> audio_token <eoa> <eos>
        squence length = 1 + 1 + 40(clip) + 1 +1 + 227(t5) +1 +1 + 750(audio) +1 +1
        """
        audio_token = audio_token.tolist()
        if len(audio_token) > max_audio_len:
            audio_token = audio_token[:max_audio_len]
        full_seq_token = (
            [self.bos_token, self.special_token_map["<bov>"]]
            + [self.special_token_map["<video>"]] * max_clip_frame
            + [self.special_token_map["<eov>"], self.special_token_map["<bot>"]]
            + [self.special_token_map["<text>"]] * max_t5_len  #
            + [self.special_token_map["<eot>"], self.special_token_map["<boa>"]]
            + audio_token
            + [self.special_token_map["<eoa>"], self.eos_token]
        )
        if len(full_seq_token) < max_len:
            full_seq_token += [self.special_token_map["<pad>"]] * (max_len - len(full_seq_token))
        full_seq_tensor = torch.tensor(full_seq_token).long()
        return full_seq_tensor

    def encode_visual_t5_audio_placeholder(self, max_len=1025, max_audio_len=750, max_clip_frame=40, max_t5_len=227):
        """ audio sequence format: <bos> <bov> clip <eov> <bot> T5 <eot> <boa> audio_token <eoa> <eos>
        squence length = 1 + 1 + 40(clip) + 1 +1 + 227(t5) +1 +1 + 750(audio) +1 +1
        """
        full_seq_token = (
            [self.bos_token, self.special_token_map["<bov>"]]
            + [self.special_token_map["<video>"]] * max_clip_frame
            + [self.special_token_map["<eov>"], self.special_token_map["<bot>"]]
            + [self.special_token_map["<text>"]] * max_t5_len  #
            + [self.special_token_map["<eot>"], self.special_token_map["<boa>"]]
            + [self.special_token_map["<audio>"]] * max_audio_len
            + [self.special_token_map["<eoa>"], self.eos_token]
        )
        if len(full_seq_token) < max_len:
            full_seq_token += [self.special_token_map["<pad>"]] * (max_len - len(full_seq_token))
        full_seq_tensor = torch.tensor(full_seq_token).long()
        return full_seq_tensor
    
    def encode_visual_t5_for_audio_infer(self, max_clip_frame=40, max_t5_len=227):
        full_seq_token = (
            [self.bos_token, self.special_token_map["<bov>"]]
            + [self.special_token_map["<video>"]] * max_clip_frame
            + [self.special_token_map["<eov>"], self.special_token_map["<bot>"]]
            + [self.special_token_map["<text>"]] * max_t5_len  #
            + [self.special_token_map["<eot>"], self.special_token_map["<boa>"]]
        )
        full_seq_tensor = torch.tensor(full_seq_token).long()
        return full_seq_tensor
