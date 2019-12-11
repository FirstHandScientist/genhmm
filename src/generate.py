import pickle
import torch
from torch import nn,distributions
import hmmlearn
from hmmlearn import hmm
from realnvp import RealNVP
import numpy as np

def generate():
    sample_sequence=20
    batchsize=10
    prior=distributions.MultivariateNormal(torch.zeros(40), torch.eye(40)) #平均と分散行列

    """modelのインポート"""
    #gmm_model=torch.load('/Users/ryusuke/Downloads/genHMM/genhmm_class61_component3/epoch20.mdl')
    gmm_model=torch.load('/Users/ryusuke/Downloads/KTH/gm_hmm/exp/gen/39feats/test/models/epoch10.mdl')
    #gmm_model1,gmm_model2=gmm_model.userdata.hmms #GenHMM()になってる
    gmm_model1=gmm_model.userdata.hmms[0]

    #sample_test=gmm_model1.networks[0][0].sample_test
    #prior=gmm_model1.networks[0][0].prior

    '''hmmモデルの準備'''
    hmm_model = hmm.MultinomialHMM(n_components = gmm_model1.n_states) #
    hmm_model.startprob_ = gmm_model1.startprob_
    hmm_model.transmat_ = gmm_model1.transmat_
    hmm_model.emissionprob_ = gmm_model1.pi

    '''生成器の決定'''
    k,s=hmm_model.sample(sample_sequence) #sが状態,kが精製機の種類 少なくとも出力は出来てる

    '''生成'''
    #z = prior.sample((batchSize, 1))
    x=[]
    print(k)
    print(s)
    count=0
    for k,s in zip(k,s):
        nn=gmm_model1.networks[s][k][0]
        #nn=gmm_model1.networks[s][k]
        z = prior.sample((1,1))
        #print(nn)
        rescale=nn.rescale
        for i in range(8):#このfor文内で処理したいのはある一時点のデータのはず
            net=nn.s[i] #ここ合ってるかな？？ちゃんと6層のNNに入れてるかな？ ちゃんと6層のNNだ'''
            z_id,z_s = z.chunk(2,dim=2) #'''ここ合ってるかな？？ xからzを作るときに、ただくっつけてるだけか確認が必要'''
            z_id_after=net(z_id) #ネットワークによって変化があるのはどこなんだろう。どんな入力を待ってるのだろう
            z_A,z_B=z_id_after.chunk(2,dim=2)
            z_A=torch.tanh(z_A)
            z_A=rescale(z_A) #rescaleは常に一緒と仮定
            z_A=z_A.exp()
            x_s=(z_s/z_A)-z_B
            x_id=z_id

            j=range(x_id.shape[2])
            x_id=x_id.detach().numpy() #numpyに直している
            x_s=x_s.detach().numpy() #numpyに直している

            if i%2 == 1:
                z=np.insert(x_s,j,x_id[:,:,j],axis=2) #ここマスクのはず
            elif i%2 == 0:
                z=np.insert(x_id,j,x_s[:,:,j],axis=2) #ここマスクのはず

            z=torch.from_numpy(z)

        count+=1
        if count==1:
            x=z
        else:
            x=torch.cat([x,z],dim=1)
    #filename='/Users/ryusuke/Downloads/data/xtest.pkl'
    filename='/Users/ryusuke/Downloads/KTH/gm_hmm/exp/gen/39feats/test/Test/xtest.pkl'
    with open(filename,'wb') as f:
        pickle.dump(x,f)

    # x=[]
    # for k,s in zip(k,s):
    #     nn = gmm_model1.networks[s][k]
    #     x_unit = sample_test(prior,nn,batchsize) #ここで一区切りが生成される
    #     x.append(x_unit) #最終的に一つの音素の連続となる
    # return x

def g(z):
    '''select the phonemes'''
    genHMM=model.userdata.hmms[0] #0は仮置き
    '''select the states'''

    '''select the networks'''
    networks=genHMM.networks #shape(3,2)ね　これのどれを使うか決めなきゃ　それ潜在状態決めた後、πで決まるよね
    for i in range(len(nn)):#iが偶数か奇数かでマスクが違う このfor文は、一回分の音素を出しているはず
        net=nn.s[i]
        rescale=nn.rescale

        z_id,z_s = z.chunk(2,dim=2)

        z_id_after=net(z_id)

        z_A,z_B=z_id_after.chunk(2,dim=2)

        z_A=torch.tanh(z_A)

        z_A=rescale(z_A)

        z_A=z_A.exp()

        x_s=(z_s/z_A)-z_B
        x_id=z_id

        j=range(x_id.shape[2])
        x_id=x_id.detach().numpy()
        x_s=x_s.detach().numpy()
        if j%2 == 1:
            z=np.insert(x_s,j,x_id[:,:,j],axis=2)
        if j%2 == 0:
            z=np.insert(x_id,j,x_s[:,:,j],axis=2)

        z=torch.from_numpy(z)

    return z
