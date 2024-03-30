
def plot(loss_values,epochs):
  # Plotting the training loss
  plt.figure(figsize=(10, 6))
  plt.plot(loss_values, 'b-o')

  plt.title("Training Loss-BERT Sentiment Analysis ")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.xticks(range(1, epochs+1))


    # Define the folder name
  folder_name = "results"

  # Check if the folder exists, if not, create it
  if not os.path.exists(folder_name):
      os.makedirs(folder_name)
      print("Folder results created.")
  else:
      print("Folder results already exists.")
  plt.savefig(os.path.join("results", 'Training_Loss.png'))
  print("[INFO] Training Loss saved to file Training_Loss.png in results folder")
  plt.show()


def train():

  text_train,labels_train=preprocess(data_split="train")
  text_validation,labels_validation=preprocess(data_split="validation")
  text_test,labels_test=preprocess(data_split="test")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  num_unique_values=3
  model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = num_unique_values,
    output_attentions = False,
    output_hidden_states = False)
  model.to(device)

  print("-"*70)
  print(model)
  print("-"*70)


  #create data loaders
  dataset_train = TextDataset(text_train, labels_train)
  dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)

  dataset_validation= TextDataset(text_validation, labels_validation)
  dataloader_validation = DataLoader(dataset_validation, batch_size=16, shuffle=True)

  dataset_test = TextDataset(text_test, labels_test)
  dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=True)


  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 1e-5, #2e-5 to 5e-5
    eps = 1e-8)
  



  # Store the average loss after each epoch so we can plot them.
  loss_values = []

  for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0

        for step, batch in enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch}")):


            # Move each component of the batch to the GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids=batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)  # Only if you're doing supervised learning
    

            model.zero_grad()

            outputs = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)
            loss = outputs[0]
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(dataloader_train)
        loss_values.append(avg_train_loss)

        print(f"Epoch {epoch} | Average Training Loss: {avg_train_loss}")

  #save model
  torch.save(model.state_dict(), f'Bert_ft.pt')
  print("[INFO] Model saved to file  Bert_ft.pt")
  plot(loss_values,epochs)


if __name__ == "__main__":
  train()

