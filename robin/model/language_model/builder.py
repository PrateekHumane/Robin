from transformers import MistralModel, LlamaModel, GPTNeoXModel
from enum import Enum

class LLMType(Enum):
    LLAMA = 'llama'
    MISTRAL = 'mistral'
    NEOX = 'neox'
    MPT = 'mpt'

def get_model_type_from_model_name(model_name: str) -> LLMType:
    model_name = model_name.lower()
    if 'mpt' in model_name:
        return LLMType.MPT
    elif 'mistral' in model_name:
        return LLMType.MISTRAL 
    elif any(x in model_name for x in ['neox', 'pythia', 'hi-nolin']):
        return LLMType.NEOX 
    else:
        return LLMType.LLAMA 
 
def get_model_type_list(cls):
    return [m.value for m in LLMType]

def build_llm_model(model_name_or_path, llm_config, llm_type=None,  **kwargs):
    # register llm_type from Enum
    if llm_type is not None:
        try:
            llm_type = LLMType(llm_type)
        except KeyError as e:
            raise ValueError(f"Invalid llm type provided {e}. Supported llm classes are {', '.join(get_model_type_list())}")
    else:
        llm_type = get_model_type_from_model_name(model_name_or_path)


    if llm_type == LLMType.MISTRAL:
        return MistralModel.from_pretrained(
            model_name_or_path,
            **llm_config
            # cache_dir=llm_config.training_args.cache_dir,
            # use_flash_attention_2 = llm_config.USE_FLASH_ATTN_2,
            # **bnb_model_from_pretrained_args
        )
    elif llm_type == LLMType.NEOX:
        return GPTNeoXModel.from_pretrained(
            model_name_or_path,
            **llm_config
            # cache_dir=training_args.cache_dir,
            # use_flash_attention_2 = USE_FLASH_ATTN_2, # The current architecture does not support Flash Attention 2.0
            # **bnb_model_from_pretrained_args
        )
    elif llm_type == LLMType.LLAMA:
        return LlamaModel.from_pretrained(
            model_name_or_path,
            **llm_config
            # cache_dir=training_args.cache_dir,
            # **bnb_model_from_pretrained_args,
            # use_flash_attention_2 = USE_FLASH_ATTN_2,
        )
    else:
        raise NotImplementedError