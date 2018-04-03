import chainer.functions as F
import chainer.links as L
import numpy as np
import chainer
import chainermn


class Discriminator(chainer.Chain):
    def __init__(self, comm, bottom_width=4, ch=512, wscale=0.02, B=100, C=5):
        w = chainer.initializers.Normal(wscale)
        self.comm = comm
        self.device = self.comm.rank
        self.B = B
        self.C = C
        super(Discriminator, self).__init__()
        if self.comm.rank == 0:
            with self.init_scope():
                self.c0_0 = L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w)
                self.c0_1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
                self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
                self.c1_1 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
                self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
                self.c2_1 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
                self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
                self.md = L.Linear(bottom_width * bottom_width * ch, B * C, initialW=w)
                self.l4 = L.Linear(bottom_width * bottom_width * ch + B, 1, initialW=w)
                self.bn0_1 = L.BatchNormalization(ch // 4, use_gamma=False)
                self.bn1_0 = L.BatchNormalization(ch // 4, use_gamma=False)
                self.bn1_1 = L.BatchNormalization(ch // 2, use_gamma=False)
                self.bn2_0 = L.BatchNormalization(ch // 2, use_gamma=False)
                self.bn2_1 = L.BatchNormalization(ch // 1, use_gamma=False)
                self.bn3_0 = L.BatchNormalization(ch // 1, use_gamma=False)

    def __call__(self, *inputs):
        if self.comm.rank == 0:
            x, = inputs
            N = x.data.shape[0]
            h = F.leaky_relu(self.c0_0(x))
            h = F.leaky_relu(self.bn0_1(self.c0_1(h)))
            h = F.leaky_relu(self.bn1_0(self.c1_0(h)))
            h = F.leaky_relu(self.bn1_1(self.c1_1(h)))
            h = F.leaky_relu(self.bn2_0(self.c2_0(h)))
            h = F.leaky_relu(self.bn2_1(self.c2_1(h)))
            feature = F.reshape(F.leaky_relu(self.c3_0(h)), (N, 8192))
            m = F.reshape(self.md(feature), (N, self.B * self.C, 1))

            m = chainermn.functions.bcast(self.comm, m, device=self.device)
            S = N // self.comm.size
            si = self.comm.rank

            m0 = F.broadcast_to(m, (N, self.B * self.C, N))
            m1 = F.transpose(m0[:, :, si*S:(si+1)*S], (2, 1, 0))
            d = F.absolute(F.reshape(m0[si*S:(si+1)*S] - m1, (S, self.B, self.C, N)))
            d = F.sum(F.exp(-F.sum(d, axis=2)), axis=2) - 1

            d = F.broadcast_to(d, (self.comm.size, S, self.B))
            ds = F.split_axis(F.transpose(d, [2, 1, 0]), self.comm.size, axis=2)
            ds = chainermn.functions.all_to_all(self.comm, ds, device=self.device)
            d = F.transpose(F.concat(ds)[:, :, 0])

            h = F.concat([feature, d])

            return self.l4(h)

        else:
            m = chainermn.functions.bcast(self.comm, None, device=self.device)
            N = m.shape[0]
            S = N // self.comm.size
            si = self.comm.rank

            m0 = F.broadcast_to(m, (N, self.B * self.C, N))
            m1 = F.transpose(m0[:, :, si*S:(si+1)*S], (2, 1, 0))
            d = F.absolute(F.reshape(m0[si*S:(si+1)*S] - m1, (S, self.B, self.C, N)))
            d = F.sum(F.exp(-F.sum(d, axis=2)), axis=2) - 1

            d = F.broadcast_to(d, (self.comm.size, S, self.B))
            ds = F.split_axis(F.transpose(d, [2, 1, 0]), self.comm.size, axis=2)
            ds = chainermn.functions.all_to_all(self.comm, ds, device=self.device)
            phi = chainer.functions.sum(ds[0])
            for d in ds[1:]:
                phi += chainer.functions.sum(d)
            return phi
