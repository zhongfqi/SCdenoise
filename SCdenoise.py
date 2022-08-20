#-*- coding: UTF-8 -*-
import os.path
import time
import scanpy as sc
import umap
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import lr_schedule
from loss import BCELossForMultiClassification
from loss import CenterLoss, VATLoss
from utils import *
from networks import *

def SCdenoise(args, source_data, target_data):
    batch_size = args.batch_size
    embedding_size = args.embedding_size
    kwargs = {'num_workers': 0, 'pin_memory': True}

    # prepare data
    class_to_idx_s = {x: i for i, x in
                    enumerate(source_data.obs['CellType'].unique())}
    source_data.obs['digit_CellType'] = source_data.obs['CellType'].map(class_to_idx_s).values
    class_to_idx_t = {x: i for i, x in
                           enumerate(target_data.obs['CellType'].unique())}
    target_data.obs['digit_CellType'] = target_data.obs['CellType'].map(class_to_idx_t).values

    train_set = {'features': source_data.X, 'labels': source_data.obs['digit_CellType']}
    test_set = {'features': target_data.X, 'labels': target_data.obs['digit_CellType']}
    print(train_set['features'].shape, test_set['features'].shape)

    data = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_set['features']), torch.LongTensor(matrix_one_hot(train_set['labels'], int(max(train_set['labels'])+1)).long()))
    source_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    data = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_set['features']), torch.LongTensor(matrix_one_hot(test_set['labels'], int(max(test_set['labels'])+1)).long())
    )
    target_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    class_num = max(train_set['labels'])+1
    # class_num_test = max(test_set['labels']) + 1

    # re-weighting the classifier
    cls_num_list = [np.sum(train_set['labels'] == i) for i in range(class_num)]
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

    # set base network
    AE_Encoder = Encoder(num_inputs=train_set['features'].shape[1], embed_size = embedding_size).cuda()
    label_classifier = LabelClassifier(AE_Encoder.output_num(), class_num).cuda()
    AE_Decoder = Decoder(AE_Encoder.output_num(), train_set['features'].shape[1]).cuda()
    total_model = nn.Sequential(AE_Encoder, label_classifier)

    center_loss = CenterLoss(num_classes=class_num, feat_dim=embedding_size, use_gpu=True)
    optimizer_centloss = torch.optim.SGD([{'params': center_loss.parameters()}], lr=0.5)

    print("output size of Encoder and LabelClassifier: ", AE_Encoder.output_num(), class_num)
    ad_net = scAdversarialNetwork(AE_Encoder.output_num(), 1024).cuda()

    # set optimizer
    config_optimizer = {"lr_type": "inv", "lr_param": {"lr": 0.001, "gamma": 0.001, "power": 0.75}}
    parameter_list = AE_Encoder.get_parameters() + ad_net.get_parameters() + label_classifier.get_parameters() + AE_Decoder.get_parameters()
    # optimizer = optim.SGD(parameter_list, lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True)
    optimizer = torch.optim.Adam(parameter_list, lr=0.0001)
    schedule_param = config_optimizer["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[config_optimizer["lr_type"]]

    # train
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    epoch_global = 0.0

    for epoch in range(args.epoch_th+1):
        # train one iter
        AE_Encoder.train(True)
        ad_net.train(True)
        label_classifier.train(True)
        AE_Decoder.train(True)

        optimizer = lr_scheduler(optimizer, epoch, **schedule_param)
        optimizer.zero_grad()
        optimizer_centloss.zero_grad()

        if epoch % len_train_source == 0:
            iter_source = iter(source_loader)
            epoch_global = epoch_global + 1
        if epoch % len_train_target == 0:
            iter_target = iter(target_loader)

        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source, labels_target = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda(), labels_target.cuda()

        feature_source = AE_Encoder(inputs_source)
        feature_target = AE_Encoder(inputs_target)

        output_source = label_classifier.forward(feature_source)
        output_target = label_classifier.forward(feature_target)

        # VAT and BNM loss
        # LDS should be calculated before the forward for cross entropy
        vat_loss = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)
        lds_loss = vat_loss(total_model, inputs_target)

        softmax_tgt = nn.Softmax(dim=1)(output_target[:, 0:class_num])
        _, s_tgt, _ = torch.svd(softmax_tgt)
        BNM_loss = -torch.mean(s_tgt)

        # domain alignment loss
        domain_prob_discriminator_1_source = ad_net.forward(feature_source)
        domain_prob_discriminator_1_target = ad_net.forward(feature_target)

        adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), \
                                                 predict_prob=domain_prob_discriminator_1_source)  # domain matching
        adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), \
                                                  predict_prob=1 - domain_prob_discriminator_1_target)
        transfer_loss = adv_loss

        # CrossEntropyLoss
        classifier_loss = nn.CrossEntropyLoss(weight=per_cls_weights)(output_source, torch.max(labels_source, dim=1)[1])

        progress = epoch / args.num_iterations
        lambd = 2 / (1 + math.exp(-10 * progress)) - 1

        dec_s = AE_Decoder(feature_source)
        dec_t = AE_Decoder(feature_target)
        recon_loss_s = nn.MSELoss()(dec_s, inputs_source)
        recon_loss_t = nn.MSELoss()(dec_t, inputs_target)
        total_loss = args.cls_coeff*classifier_loss + + lambd*args.alpha*lds_loss\
                     + lambd*args.DA_coeff * transfer_loss + lambd*args.BNM_coeff*BNM_loss \
                     + args.mse_coeff*(recon_loss_s + recon_loss_t)

        total_loss.backward()
        optimizer.step()

        if epoch % 1000 == 0 and epoch != 0:
            with torch.no_grad():
                AE_Encoder.eval()
                AE_Decoder.eval()
                feature_target = AE_Encoder(torch.FloatTensor(test_set['features']).cuda())
                output_target = AE_Decoder.forward(feature_target)

                # get imputed traget matrix
                imputed_target_x = output_target.cpu().numpy()
                # write to h5ad
                output_adata = sc.AnnData(imputed_target_x)
                output_adata.obs_names = target_data.obs_names
                output_adata.var_names = target_data.var_names
                output_adata.obs['CellType'] = target_data.obs['CellType']
                output_adata.write_h5ad(os.path.join(args.result_path, 'denoised_'+args.dataset+'epoch_'+str(epoch)+'.h5ad'))
