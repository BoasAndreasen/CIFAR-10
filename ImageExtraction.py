import tensorflow as tf



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
## b"data"
## b"labels"
print(len(unpickle("cifar-10-batches-py/data_batch_1")[b"data"][0]))

full_dataset = {b"data":[],b"labels":[]}
for i in range(1,6):
    extracted_data = unpickle("cifar-10-batches-py/data_batch_"+str(i))
    full_dataset[b"data"] += list(extracted_data[b"data"])
    full_dataset[b"labels"] += list(extracted_data[b"labels"])

print(len(full_dataset[b"data"]))

##def convert_to_grey_scale(dataset):
##    output = {}
##    output[b"labels"] = dataset[b"labels"]
##    output[b"data"] = []
##    total = len(output[b"labels"])
##    print("Total to Convert: "+str(total))
##    print("Progress:",end = "")
##    counter = 0
##    for image in dataset[b"data"]:
##        temp = []
##        for pix in range(0,1024):
##            temp.append(int((image[pix]+image[pix+1024]+image[pix+2048])/3))
##        output[b"data"] += [temp]
##        counter +=1
##        if counter%(total/100) == 0:
##            print('#',end='')
##  return output

##grey = convert_to_grey_scale(full_dataset)
