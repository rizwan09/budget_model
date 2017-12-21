import theano
import theano.tensor as T
import numpy

c = theano.tensor.vector("c")
x = T.scalar("x")
embs = theano.shared([ [[1,2, 3, 4], [2, 3, 4, 5]], [[1,2, 3, 4], [2, 3, 4, 5]] ])
def get_avg_emb(  i, embs_mid,):
        return i#T.mean(embs_mid[i-self.nneighbor: i+self.nneighbor+1,:,:], axis = 0)#theano.scan_module.until(embs_mid)

avg_emb_middle, _ = theano.scan(
                    fn = get_avg_emb,
                    sequences = T.arange( 2),
                    non_sequences = embs,
                    outputs_info = None
                )
print avg_emb_middle.eval()
