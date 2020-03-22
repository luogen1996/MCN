# encoding=utf8
# %matplotlib inline
import numpy as np
import os
from refer import REFER
import os.path as osp
import cv2
import argparse
parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--data_root',  type=str) # contains refclef, refcoco, refcoco+, refcocog and images
parser.add_argument('--output_dir',  type=str)
parser.add_argument('--dataset', type=str, choices=['refcoco', 'refcoco+','refcocog','refclef'],default='refcoco')
parser.add_argument('--split',  type=str,default='umd')
parser.add_argument('--generate_mask',  action='store_true')
args = parser.parse_args()
# data_root # contains refclef, refcoco, refcoco+, refcocog and images

refer = REFER(args.data_root, args.dataset, args.split)

print ('dataset [%s_%s] contains: ' % (args.dataset, args.split))
ref_ids = refer.getRefIds()
image_ids = refer.getImgIds()
print ('%s expressions for %s refs in %s images.' % (len(refer.Sents), len(ref_ids), len(image_ids)))

print('\nAmong them:')
if args.dataset == 'refclef':
    if args.split == 'unc':
        splits = ['train', 'val', 'testA','testB','testC']
    else:
        splits = ['train', 'val', 'test']
elif args.dataset == 'refcoco':
    splits = ['train', 'val', 'testA', 'testB']
elif args.dataset == 'refcoco+':
    splits = ['train', 'val',  'testA', 'testB']
elif args.dataset == 'refcocog':
    splits = ['train', 'val', 'test']  # we don't have test split for refcocog right now.


# split data as a type in splits list
for split in splits:
    ref_ids = refer.getRefIds(split=split)
    print('%s refs are in split [%s].' % (len(ref_ids), split))


# show a batch data with bounding box,cat,sentences
def show_a_batch(batch_size):
    split='train'
    # batch_size=32
    ref_ids = refer.getRefIds(split=split)
    print(split+'_size:',np.alen(ref_ids))
    batch_index=list(np.random.choice(np.alen(ref_ids),batch_size))

    # print(refer.Refs)
    ref_id = [ref_ids[i] for i in batch_index]
    refs = [refer.Refs[i] for i in ref_id]
    bboxs=[refer.getRefBox(i) for i in ref_id]
    sentences=[ref['sentences'] for ref in refs]
    image_urls=[refer.loadImgs(image_ids=ref['image_id']) for ref in refs]
    cats = [refer.loadCats(cat_ids=ref['category_id']) for ref in refs]
    # plt.figure()
    # plt.subplot(batch_size)
    grid_width = 2
    grid_height = int(batch_size / grid_width)
    # fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width*10, 10*grid_height))
    for i in range(batch_size):
        print('bbox for batch[{}]:'.format(i),bboxs[i])
        print('sentences for batch[{}]:'.format(i))
        for sid, sent in enumerate(sentences[i]):
            print('%s. %s' % (sid+1, sent['sent']))
        print('cats for batch[{}]:'.format(i), cats[i])

        image_url=image_urls[i][0]
        image=cv2.imread(osp.join(refer.IMAGE_DIR, image_url['file_name']))
        print(image.shape)
        # print(bboxs[i][0])
        cv2.rectangle(image,(int(bboxs[i][0]), int(bboxs[i][1])), (int(bboxs[i][0]+bboxs[i][2]),int(bboxs[i][1]+ bboxs[i][3])),255,3)
        cv2.putText(image,
                        str(sent['sent']),
                        (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .9,(0,255,0), 2)
        os.mkdir('debug_vis')
        cv2.imwrite('./debug_vis/'+image_url['file_name'], image)
        cv2.imwrite('./debug_vis/mask'+image_url['file_name'], refer.getMask(refs[i])['mask']*255)
        # ax.imshow(image)
    # plt.show()
def cat_process(cat):
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    return  cat
def bbox_process(bbox,cat,segement_id):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    box_info = " %d,%d,%d,%d,%d,%d" % (int(x_min), int(y_min), int(x_max), int(y_max), int(cat),int(segement_id))
    return box_info
def prepare_dataset(dataset,splits,output_dir,generate_mask=False):
    # split_type='train'
    # splits=[split_type]
    # batch_size=32
    if not os.path.exists(os.path.join(output_dir,'anns',dataset)):
        os.makedirs(os.path.join(output_dir,'anns',dataset))
    if not os.path.exists(os.path.join(output_dir,'masks',dataset)):
        os.makedirs(os.path.join(output_dir,'masks',dataset))
    for split in splits:
        f = open(os.path.join(output_dir,'anns',dataset,split + '.txt'), 'w')
        # print(split)
        split_num=0
        ll=0
        ref_ids = refer.getRefIds(split=split)
        print(split+'_size:',np.alen(ref_ids))
        for i in ref_ids:
            # ref_id = ref_ids[i]
            refs = refer.Refs[i]
            bboxs=refer.getRefBox(i)
            sentences=refs['sentences']
            image_urls=refer.loadImgs(image_ids=refs['image_id'])[0]
            cat = cat_process(refs['category_id'])
            image_urls=image_urls['file_name']
            if dataset=='refclef' and  image_urls in ['19579.jpg', '17975.jpg', '19575.jpg']:
                continue
            box_info=bbox_process(bboxs,cat,i) #add segement id
            f.write(image_urls)
            f.write(box_info)
            # f.write(' '+str(i))
            if generate_mask:
                np.save(os.path.join(output_dir,'masks',dataset,str(i)+'.npy'),refer.getMask(refs)['mask'])  #if need seg mask ,set it!
            for sentence in sentences:
                f.write(' ~ ')
                # print(sentence['sent'].encode('UTF-8'))
                f.write(sentence['sent'])
                if ll<len(sentence['sent']):
                    ll=len(sentence['sent'])
            f.write('\n')
            split_num+=1
        print('split_num:',split_num)
        print('max_len:',ll)
        f.close()

def prepare_sentences_refcoco():
    splits=['train','val']
    # batch_size=32
    f = open('sentences.txt', 'w')
    for split in splits:
        print(split)
        ref_ids = refer.getRefIds(split=split)
        print(split+'_size:',np.alen(ref_ids))
        for i in range(np.alen(ref_ids)):
            refs = refer.Refs[i]
            sentences=refs['sentences']
            for sentence in sentences:
                f.write(sentence['sent'])
                f.write('\n')
    f.close()
def test_length():
    max_len=0
    word_l_count=np.zeros([50],dtype=np.int)
    with open('./refcocog/train.txt') as f:
        lines = f.readlines()
        for j in range(len(lines)):
            line=lines[j].split()
            stop = len(line)
            for i in range(1, len(line)):
                if (line[i] == '~'):
                    stop = i
                    break
            sentences = []
            sent_stop = stop + 1
            for i in range(stop + 1, len(line)):
                if line[i] == '~':
                    # sentences.append(line[sent_stop:i])
                    # print(len(line[sent_stop:i]))
                    word_l_count[len(line[sent_stop:i])]+=1
                    # if len(line[sent_stop:i])>max_len:
                    #     max_len=len(line[sent_stop:i])
                    sent_stop = i + 1
    for i in range(50):
        if word_l_count[i]>0:
            print('length:%d'%i,',count:%d'%word_l_count[i])
    # print('max_len:',max_len)
        # print(len(lines))


prepare_dataset(args.dataset,splits,args.output_dir,args.generate_mask)
