from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import json


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
dataset = load_dataset('seamew/chnsenticorp')

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

encoded_dataset = dataset.map(tokenize_function, batched=True)


if __name__ == "__main__":
    with open('./config/config_00.json', 'r') as f:
        config = json.load(f)
    training_args = TrainingArguments(**config['training_args'])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['test'],
    )

    trainer.train()

    result = trainer.evaluate(eval_dataset=encoded_dataset['test'])
    print(result)

    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')
