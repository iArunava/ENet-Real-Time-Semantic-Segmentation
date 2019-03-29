import torch
import torch.nn as nn
from utils import *
from models.ENet import ENet
import sys
from tqdm import tqdm

def train(FLAGS):

    # Defining the hyperparameters
    device =  FLAGS.cuda
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    lr = FLAGS.learning_rate
    print_every = FLAGS.print_every
    eval_every = FLAGS.eval_every
    save_every = FLAGS.save_every
    nc = FLAGS.num_classes
    wd = FLAGS.weight_decay
    ip = FLAGS.input_path_train
    lp = FLAGS.label_path_train
    ipv = FLAGS.input_path_val
    lpv = FLAGS.label_path_val
    print ('[INFO]Defined all the hyperparameters successfully!')
    
    # Get the class weights
    print ('[INFO]Starting to define the class weights...')
    pipe = loader(ip, lp, batch_size='all')
    class_weights = get_class_weights(pipe, nc)
    print ('[INFO]Fetched all class weights successfully!')

    # Get an instance of the model
    enet = ENet(nc)
    print ('[INFO]Model Instantiated!')
    
    # Move the model to cuda if available
    enet = enet.to(device)

    # Define the criterion and the optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    optimizer = torch.optim.Adam(enet.parameters(),
                                 lr=lr,
                                 weight_decay=wd)
    print ('[INFO]Defined the loss function and the optimizer')

    # Training Loop starts
    print ('[INFO]Staring Training...')
    print ()

    train_losses = []
    eval_losses = []
    
    # Assuming we are using the CamVid Dataset
    bc_train = 367 // batch_size
    bc_eval = 101 // batch_size

    pipe = loader(ip, lp, batch_size)
    eval_pipe = loader(ipv, lpv, batch_size)

    epochs = epochs
            
    for e in range(1, epochs+1):
            
        train_loss = 0
        print ('-'*15,'Epoch %d' % e, '-'*15)
        
        enet.train()
        
        for _ in tqdm(range(bc_train)):
            X_batch, mask_batch = next(pipe)
            
            #assert (X_batch >= 0. and X_batch <= 1.0).all()
            
            X_batch, mask_batch = X_batch.to(device), mask_batch.to(device)

            optimizer.zero_grad()

            out = enet(X_batch.float())

            loss = criterion(out, mask_batch.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            
        print ()
        train_losses.append(train_loss)
        
        if (e+1) % print_every == 0:
            print ('Epoch {}/{}...'.format(e, epochs),
                    'Loss {:6f}'.format(train_loss))
        
        if e % eval_every == 0:
            with torch.no_grad():
                enet.eval()
                
                eval_loss = 0
                
                for _ in tqdm(range(bc_eval)):
                    inputs, labels = next(eval_pipe)

                    inputs, labels = inputs.to(device), labels.to(device)
                    out = enet(inputs)
                    
                    loss = criterion(out, labels.long())

                    eval_loss += loss.item()

                print ()
                print ('Loss {:6f}'.format(eval_loss))
                
                eval_losses.append(eval_loss)
            
        if e % save_every == 0:
            checkpoint = {
                'epochs' : e,
                'state_dict' : enet.state_dict()
            }
            torch.save(checkpoint, './ckpt-enet-{}-{}.pth'.format(e, train_loss))
            print ('Model saved!')

        print ('Epoch {}/{}...'.format(e+1, epochs),
               'Total Mean Loss: {:6f}'.format(sum(train_losses) / epochs))

    print ('[INFO]Training Process complete!')
