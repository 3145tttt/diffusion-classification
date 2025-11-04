from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate


def get_metric(metric_type):
    assert metric_type == "accuracy", f"{metric_type}"
    accuracy = evaluate.load(metric_type)
    return accuracy


def compute_metrics(eval_pred, metric):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def get_args(train_conf, is_bf16):
    training_args = TrainingArguments(
        output_dir=train_conf.run_name,
        remove_unused_columns=False,
        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="best",
        learning_rate=train_conf.lr,
        per_device_train_batch_size=train_conf.batch_size,
        gradient_accumulation_steps=train_conf.gradient_accumulation_steps,
        per_device_eval_batch_size=train_conf.eval_batch_size,
        max_steps=train_conf.max_steps,
        warmup_ratio=train_conf.warmip_ratio,
        eval_steps=train_conf.eval_freq,
        logging_steps=train_conf.log_freq,
        save_steps=train_conf.save_freq,
        bf16=is_bf16,
        fp16=not is_bf16,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to=train_conf.report_to,
        dataloader_num_workers=train_conf.dataloader_num_workers
    )

    return training_args

def create_trainer(
    model, 
    training_args, 
    data_collator,
    train_dataset,
    eval_dataset,
    metric
):    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda t: compute_metrics(t, metric),
    )

    return trainer