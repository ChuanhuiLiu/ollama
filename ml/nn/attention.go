package nn

import (
	"fmt"

	"github.com/ollama/ollama/ml"
)

// Attention implements scaled dot-product attention for transformer models:
// Attention(Q, K, V) = softmax(QK^T/√d_k)V
//
// Parameters:
//   - ctx: Context for tensor operations
//   - query: Query tensor (Q) with shape [d_k, heads, seq_len_q]
//   - key: Key tensor (K) with shape [d_k, kv_heads, seq_len_k]
//   - value: Value tensor (V) with shape [d_v, kv_heads, seq_len_k]
//   - mask: Optional attention mask that is added to the attention score. If
//     provided, should broadcast to [seq_len_k, seq_len_q, heads]
//   - scale: Scaling factor, typically 1/√d_k where d_k is the key dimension
//
// Returns:
//
//	Attention output with shape [d_v, heads, seq_len_q]
func Attention(ctx ml.Context, query, key, value, mask ml.Tensor, scale float64) ml.Tensor {
	if query.Dim(0) != key.Dim(0) {
		panic(fmt.Errorf("d_k in attention operation does not match between query(%v) and key(%v)", query.Dim(0), key.Dim(0)))
	}

	if mask != nil && query.Dim(2) != mask.Dim(1) {
		panic(fmt.Errorf("seq_len_q in attention operation does not match between query(%v) and mask(%v)", query.Dim(2), mask.Dim(1)))
	}

	if key.Dim(1) != value.Dim(1) {
		panic(fmt.Errorf("kv_heads in attention operation does not match between key(%v) and value(%v)", key.Dim(1), value.Dim(1)))
	}

	if key.Dim(2) != value.Dim(2) {
		panic(fmt.Errorf("seq_len_k in attention operation does not match between key(%v) and value(%v)", key.Dim(2), value.Dim(2)))
	}

	if mask != nil && key.Dim(2) != mask.Dim(0) {
		panic(fmt.Errorf("seq_len_k in attention operation does not match between key(%v) and mask(%v)", key.Dim(2), mask.Dim(0)))
	}

	if sdpa, ok := query.(ml.ScaledDotProductAttention); ok {
		return sdpa.ScaledDotProductAttention(ctx, key, value, mask, scale)
	} else {
		query = query.Permute(ctx, 0, 2, 1, 3)
		key = key.Permute(ctx, 0, 2, 1, 3)
		value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

		kq := key.MulmatFullPrec(ctx, query)

		kq = kq.Scale(ctx, scale)
		if mask != nil {
			kq = kq.Add(ctx, mask)
		}
		kq = kq.Softmax(ctx)

		kqv := value.Mulmat(ctx, kq)
		return kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	}
}
