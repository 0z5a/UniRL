# Modified from Janus-Pro
# https://github.com/deepseek-ai/Janus/blob/main/janus/utils/conversation.py

from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import IntEnum, auto
from copy import deepcopy

class SeparatorStyle(IntEnum):
    ADD_COLON_SPACE_SINGLE = auto()
    NONE = auto()
    ADD_ENTER_SINGLE = auto()


@dataclass
class Conversation(object):
    name: str
    system_template: str = "{system_message}"
    system_message: str = ""
    roles: Tuple[str, str] = ("User", "Assistant")
    messages: List[List[str]] = ()
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SPACE_SINGLE
    sep: str = "\n"
    sep2: str = None
    sep_sp: str = None
    stop_token_ids: list[int] = None
    pretrain_roles: Tuple[str, str] = ("", "")
    pretrain_sep: str = ""
    pretrain_sep2: str = ""
    pretrain_sep_sp: str = ""
    add_pad: bool = False    # for encode general, instead of fix those choise in apply_general_template, we can add these flags here
    add_bos: bool = True
    add_eos: bool = False
    use_answer: bool = True

    def get_prompt(self, return_type="str", add_system=True):
        system_prompt = self.system_template.format(system_message=self.system_message)
        prompt_list = []

        if self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            seps = [self.sep, self.sep2]
            if add_system:
                prompt_list.append(("System", system_prompt + self.sep_sp if system_prompt else ""))
            for i, (role, message) in enumerate(self.messages):
                if message:
                    prompt_list.append((role, f"{role}: {message}{seps[i % 2]}"))
                else:
                    prompt_list.append((role, f"{role}: "))

        elif self.sep_style == SeparatorStyle.NONE:
            seps = [self.sep, self.sep2]
            if add_system:
                prompt_list.append(("System", system_prompt + self.sep_sp if system_prompt else ""))
            for i, (role, message) in enumerate(self.messages):
                if message:
                    prompt_list.append((role, f"{role}{message}{seps[i % 2]}"))
                else:
                    prompt_list.append((role, f"{role}"))
        elif self.sep_style == SeparatorStyle.ADD_ENTER_SINGLE:
            seps = [self.sep, self.sep2]
            if add_system:
                prompt_list.append(("System", system_prompt + self.sep_sp if system_prompt else ""))
            for i, (role, message) in enumerate(self.messages):
                if message:
                    prompt_list.append((role, f"{role}\n{message}{seps[i % 2]}"))
                else:
                    prompt_list.append((role, f"{role}\n"))
        else:
            raise NotImplementedError(f"Unsupported sep_style: {self.sep_style}")

        if return_type == "str":
            prompt = "".join([msg for _, msg in prompt_list])
        else:
            prompt = prompt_list

        return prompt

    def get_role_prefix(self, role):
        if role == "":
            return ""
        if self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            return f"{role}: "
        elif self.sep_style == SeparatorStyle.NONE:
            return f"{role}"
        elif self.sep_style == SeparatorStyle.ADD_ENTER_SINGLE:
            return f"<|im_start|>{role}\n"
        else:
            raise NotImplementedError(f"Unsupported sep_style: {self.sep_style}")

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def add_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def copy(self):
        return deepcopy(self)

    def empty(self, name=None):
        """Return an empty conversation with the same template."""
        return Conversation(
            name=name or self.name,
            system_template=self.system_template,
            system_message="",
            roles=self.roles,
            messages=[],
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            sep_sp=self.sep_sp,
            stop_token_ids=self.stop_token_ids,
            pretrain_roles=self.pretrain_roles,
            pretrain_sep=self.pretrain_sep,
            pretrain_sep2=self.pretrain_sep2,
            pretrain_sep_sp=self.pretrain_sep_sp,
            add_pad=self.add_pad,
            add_bos=self.add_bos,
            add_eos=self.add_eos,
            use_answer=self.use_answer,
        )


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template


register_conv_template(
    Conversation(
        name="hunyuan-gemini-alpha",
        system_template="{system_message}",
        system_message="",
        roles=("User", "Assistant"),
        messages=[],
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n\n",
        sep2="<|endoftext|>",
        sep_sp="\n\n",
        stop_token_ids=[127957],
    )
)

register_conv_template(
    Conversation(
        name="hunyuan-dense-7b-gemini",
        system_template="{system_message}",
        system_message="",
        roles=("User", "Assistant"),
        messages=[],
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n\n",
        sep2="<|endoftext|>",
        sep_sp="\n\n",
        stop_token_ids=[127957],
    )
)

register_conv_template(
    conv_templates["hunyuan-dense-7b-gemini"].empty(
        name="hunyuan-dense-3b"
    )
)

register_conv_template(
    conv_templates["hunyuan-dense-7b-gemini"].empty(
        name="hunyuan-moe-a13b-gemini"
    )
)

register_conv_template(
    Conversation(
        name="hunyuan-moe-a3b",
        system_template="{system_message}",
        system_message="",
        roles=("<｜hy_User｜>", "<｜hy_Assistant｜>"),
        messages=[],
        sep_style=SeparatorStyle.NONE,
        sep="",
        sep2="<｜hy_place▁holder▁no▁2｜>",
        sep_sp="<｜hy_place▁holder▁no▁3｜>",
        stop_token_ids=[120020],
    )
)

register_conv_template(
    conv_templates["hunyuan-moe-a3b"].empty(
        name="hunyuan-moe-a3b-gemini"
    )
)

register_conv_template(
    Conversation(
        name="hunyuan-moe-a3b-vlm",
        system_template="{system_message}",
        system_message="",
        roles=("", ""),
        messages=[],
        sep_style=SeparatorStyle.NONE,
        sep="<｜hy_User｜>",
        sep2="<｜hy_Assistant｜><｜hy_place▁holder▁no▁2｜>",
        sep_sp="<｜hy_place▁holder▁no▁3｜>",
        stop_token_ids=[120007, 120020],
        pretrain_sep="<｜hy_User｜>",
        pretrain_sep2="<｜hy_Assistant｜><｜hy_place▁holder▁no▁2｜>",
        pretrain_sep_sp="<｜hy_place▁holder▁no▁3｜>",
    )
    # Notice that hunyuan-vlm uses post prefix: i.e., the role names are added in the sep and sep2.
)

register_conv_template(
    conv_templates["hunyuan-moe-a3b-vlm"].empty(
        name="hunyuan-moe-a3b-vlm-gemini"
    )
)
register_conv_template(
    conv_templates["hunyuan-moe-a3b-vlm"].empty(
        name="hunyuan-moe-a3b-vlm-gemini-video"
    )
)

register_conv_template(
    Conversation(
        name="hunyuan3-moe-a3b-vlm",
        system_template="{system_message}",
        system_message="",
        roles=("<｜hy_User｜>", "<｜hy_Assistant｜>"),
        messages=[],
        sep_style=SeparatorStyle.NONE,
        sep="",
        sep2="<｜/hy_Assistant｜><｜hy_▁eod▁｜>",
        sep_sp="<｜hy_place▁holder▁no▁3｜>",
        stop_token_ids=[120025, 120020],
        pretrain_sep="",
        pretrain_sep2="<｜/hy_Assistant｜><｜hy_▁eod▁｜>",
        pretrain_sep_sp="<｜hy_place▁holder▁no▁3｜>",
    )
)

register_conv_template(
    conv_templates["hunyuan3-moe-a3b-vlm"].empty(
        name="hunyuan3-moe-a3b-gemini"
    )
)


register_conv_template(
    Conversation(
        name="hymm-v3-5",
        system_template="{system_message}",
        system_message="",
        roles=("<｜hy_User｜>", "<｜hy_Assistant｜>"),
        messages=[],
        sep_style=SeparatorStyle.NONE,
        sep="",
        sep2="<eos:6124c78e>",
        sep_sp="",
        stop_token_ids=[120020, 120025],
        pretrain_sep="",
        pretrain_sep2="<｜hy_place▁holder▁no▁2｜>",   # EOD, only for pretrain
        pretrain_sep_sp="",
        use_answer=False,
    )
)

register_conv_template(
    conv_templates["hymm-v3-5"].empty(
        name="hunyuan-moe-a3b-vlm-gemini-mot-gen-a5b"
    )
)

register_conv_template(
    Conversation(
        name="hunyuan3-hymm-v3-5",
        system_template="{system_message}",
        system_message="",
        roles=("<｜hy_User｜>", "<｜hy_Assistant｜>"),
        messages=[],
        sep_style=SeparatorStyle.NONE,
        sep="",
        sep2="<｜/hy_Assistant｜>",
        sep_sp="",
        stop_token_ids=[120020, 120025],
        pretrain_sep="",
        pretrain_sep2="<｜/hy_Assistant｜>",
        pretrain_sep_sp="",
        use_answer=False,
    )
)

register_conv_template(
    conv_templates["hunyuan3-hymm-v3-5"].empty(
        name="hunyuan3-moe-a3b-gemini-mot-gen-a10b"
    )
)

register_conv_template(
    Conversation(
        name="qwen-vl-30b-a3b-instruct",
        system_template="{system_message}",
        system_message="",
        roles=("user", "assistant"),
        messages=[],
        sep_style=SeparatorStyle.ADD_ENTER_SINGLE,
        sep="<|im_end|>\n",
        sep2="<|im_end|>",
        sep_sp="\n\n",
        stop_token_ids=[151643, 151645],
        add_bos=False,
    )
)

register_conv_template(
    Conversation(
        name="qwen-vl-v2-li-dit",
        system_template="{system_message}",
        system_message="",
        roles=("", ""),
        messages=[],
        sep_style=SeparatorStyle.NONE,
        sep="",
        sep2="",
        sep_sp="",
        stop_token_ids=[151645],
        add_bos=False,
        use_answer=False,
    )
)

def get_conversation_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()
