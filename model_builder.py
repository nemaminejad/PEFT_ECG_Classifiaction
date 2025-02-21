
from transformers import AutoImageProcessor, BitsAndBytesConfig, AutoModelForImageClassification
import torch
from peft import LoraConfig, get_peft_model,TaskType, prepare_model_for_kbit_training

# getattr



def setup_tokenizer(model_checkpoint):
  """the image processor """
  # Setup tokenizer
  image_processor = AutoImageProcessor.from_pretrained(
                    model_checkpoint,device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
                    trust_remote_code=False) # prevents running custom model files on your machine)

  return image_processor

def setup_model(model_checkpoint,label2id, id2label,batch_size= 4):
  """
  Initializes and configures a quantized image classification model

  This function:
  - Loads an image processor with appropriate padding and token settings.
  - Sets up quantization using 4-bit precision for efficient training.
  - Configures  for parameter-efficient fine-tuning.
  - Loads a pre-trained image classification model with label mappings.
  - Enables gradient checkpointing and prepares the model for k-bit training.
  - Returns the quantized  model

  """
  # Quantization config
  compute_dtype = getattr(torch, "float16")
  quant_config = BitsAndBytesConfig(
      load_in_4bit=True, #4-bit precision
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_quant_storage=torch.bfloat16,
      bnb_4bit_use_double_quant=True,
  )

  # Identify the model
  batch_size = batch_size # batch size for training and evaluation

  model = AutoModelForImageClassification.from_pretrained(
      model_checkpoint,
      label2id=label2id,
      id2label=id2label,
      num_labels = 5,
      quantization_config=quant_config, #Qunatization
      torch_dtype=torch.bfloat16,
      # low_cpu_mem_usage=True,
      # device_map="auto",
      ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
  )

  model.config.use_cache = False
  model.config.pretraining_tp = 1
  model.train() # model in training mode (dropout modules are activated)

  # enable gradient check pointing
  model.gradient_checkpointing_enable()

  # enable quantized training
  model = prepare_model_for_kbit_training(model)
  # print_trainable_parameters(model) # in order to check number of trainable parameters


  return model
  
def apply_lora_to_model(config, model):

 # LORA config
  lora_config = LoraConfig(
      r=config['r'],
      lora_alpha=config['r'],
      target_modules=config['target_modules']+['classifier'],
      lora_dropout= config['lora_dropout'],
      bias="none",
      modules_to_save=["classifier"],
      # task_type=TaskType.SEQ_CLASSIFICATION
  )

  lora_model = get_peft_model(model, lora_config)

  return lora_model