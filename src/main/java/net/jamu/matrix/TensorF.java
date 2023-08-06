/*
 * Copyright 2023 Stefan Zobel
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.jamu.matrix;

import java.util.Arrays;

import net.dedekind.blas.Blas;
import net.frobenius.TTrans;

/**
 * A {@code TensorF} is a 3-dimensional stack of 2-dimensional dense matrices of
 * primitive floats with column-major storage layout. A Tensor of
 * {@code depth = 1} is a single matrix. The addressing is zero based. All
 * operations throw a {@code NullPointerException} if any of the method
 * arguments is {@code null}.
 * 
 * @since 1.4.0
 */
public class TensorF extends TensorBase {

    private static final float BETA = 1.0f;
    private static final int OFFS = 0;

    protected float[] a;

    /**
     * Create a new {@code TensorF} of matrix dimension {@code (rows, cols)}
     * which can hold a single matrix.
     * 
     * @param rows
     *            number of matrix rows
     * @param cols
     *            number of matrix columns
     */
    public TensorF(int rows, int cols) {
        this(rows, cols, 1);
    }

    /**
     * Create a new {@code TensorF} of matrix dimension {@code (rows, cols)}
     * which holds {@code depth} matrices.
     * 
     * @param rows
     *            number of matrix rows
     * @param cols
     *            number of matrix columns
     * @param depth
     *            the number of matrices in this tensor
     */
    public TensorF(int rows, int cols, int depth) {
        super(rows, cols, depth);
        a = new float[length];
    }

    /**
     * Create a new {@code TensorF} from the passed {@code MatrixF}. The
     * underlying storage of the argument gets copied.
     * 
     * @param A
     *            the matrix from which the tensor should be constructed
     */
    public TensorF(MatrixF A) {
        super(A.numRows(), A.numColumns(), 1);
        a = Arrays.copyOf(A.getArrayUnsafe(), A.getArrayUnsafe().length);
    }

    /**
     * Create a new {@code TensorF} from the passed {@code TensorF}. The
     * underlying storage of the argument gets copied.
     * 
     * @param A
     *            the tensor from which the new tensor should be constructed
     */
    public TensorF(TensorF A) {
        super(A.rows, A.cols, A.depth);
        a = Arrays.copyOf(A.a, A.a.length);
    }

    /**
     * Set the value at {@code (row, col)} in the matrix at position
     * {@code layer} to {@code val}.
     * 
     * @param row
     *            row index, zero-based
     * @param col
     *            column index, zero-based
     * @param layer
     *            matrix index, zero-based
     * @param val
     *            new value
     * @return this tensor (mutated)
     */
    public TensorF set(int row, int col, int layer, float val) {
        checkIndex(row, col, layer);
        return setUnsafe(row, col, layer, val);
    }

    /**
     * Get the matrix element at {@code (row, col)} from the matrix at position
     * {@code layer}.
     * 
     * @param row
     *            row index, zero-based
     * @param col
     *            column index, zero-based
     * @param layer
     *            matrix index, zero-based
     * @return the matrix element at {@code (row, col)}
     */
    public float get(int row, int col, int layer) {
        checkIndex(row, col, layer);
        return getUnsafe(row, col, layer);
    }

    /**
     * Set the value at {@code (row, col)} in the matrix at position
     * {@code layer} to {@code val} without any bounds checking.
     * 
     * @param row
     *            row index, zero-based
     * @param col
     *            column index, zero-based
     * @param layer
     *            matrix index, zero-based
     * @param val
     *            new value
     * @return this tensor (mutated)
     */
    public TensorF setUnsafe(int row, int col, int layer, float val) {
        a[idx(row, col, layer)] = val;
        return this;
    }

    /**
     * Get the matrix element at {@code (row, col)} from the matrix at position
     * {@code layer} without any bounds checking.
     * 
     * @param row
     *            row index, zero-based
     * @param col
     *            column index, zero-based
     * @param layer
     *            matrix index, zero-based
     * @return the matrix element at {@code (row, col)}
     */
    public float getUnsafe(int row, int col, int layer) {
        return a[idx(row, col, layer)];
    }

    /**
     * Set the content of the matrix at position {@code layer} to the passed
     * matrix {@code B}. The underlying storage of the argument gets copied.
     * 
     * @param B
     *            the matrix whose values should be used for the values of the
     *            matrix at position {@code layer}
     * @param layer
     *            the index for the layer
     * @return this tensor (mutated)
     */
    public TensorF set(MatrixF B, int layer) {
        Checks.checkEqualDimension(this, B);
        int start = startIdx(layer);
        float[] _b = B.getArrayUnsafe();
        System.arraycopy(_b, 0, a, start, _b.length);
        return this;
    }

    /**
     * Get the content of the matrix at position {@code layer} as a
     * {@code MatrixF}.
     * 
     * @param layer
     *            the layer index for matrix to retrieve
     * @return a newly constructed {@code MatrixF} that represents the content
     *         at the index {@code layer}
     */
    public MatrixF get(int layer) {
        int start = startIdx(layer);
        int len = stride();
        float[] _a = new float[len];
        System.arraycopy(a, start, _a, 0, len);
        return new SimpleMatrixF(rows, cols, _a);
    }

    /**
     * Append a single additional matrix layer after the current last layer with
     * the contents of the supplied {@code MatrixF}. The underlying storage of
     * the argument gets copied and the depth of this tensor increases by one.
     * 
     * @param B
     *            the matrix whose values should be used for the values of the
     *            new layer
     * @return this tensor (mutated)
     */
    public TensorF append(MatrixF B) {
        Checks.checkEqualDimension(this, B);
        float[] tmp = growAndCopyForAppend(B);
        float[] _b = B.getArrayUnsafe();
        System.arraycopy(_b, 0, tmp, length, _b.length);
        a = tmp;
        length = tmp.length;
        ++depth;
        return this;
    }

    /**
     * {@code C = alpha * A * B + C} where {@code A} is this tensor. On exit,
     * the tensor {@code C} is overwritten by the result of the operation. If
     * there is a mismatch between the depths of the participating tensors the
     * shortest depth is chosen to reduce the operation to a common denominator
     * (in which case the excess layers of the longer tensors are left
     * untouched).
     * 
     * @param alpha
     *            scalar scale factor for the multiplication
     * @param B
     *            tensor to be multiplied from the right
     * @param C
     *            the tensor to add on input, contains the result of the
     *            operation on output
     * @return {@code C}
     */
    public TensorF multAdd(float alpha, TensorF B, TensorF C) {
        Checks.checkMultAdd(this, B, C);
        int _depth = Math.min(Math.min(this.depth, B.depth), C.depth);
        Blas blas = Matrices.getBlas();
        blas.sgemm_multi(TTrans.NO_TRANS.val(), TTrans.NO_TRANS.val(), C.numRows(), C.numColumns(), cols, alpha, a,
                OFFS, Math.max(1, rows), B.getArrayUnsafe(), OFFS, Math.max(1, B.numRows()), BETA, C.getArrayUnsafe(),
                OFFS, Math.max(1, C.numRows()), _depth, this.stride(), B.stride(), C.stride());
        return C;
    }

    /**
     * <code>C = alpha * A<sup>T</sup> * B + C</code> where {@code A} is this
     * tensor. On exit, the tensor {@code C} is overwritten by the result of the
     * operation. If there is a mismatch between the depths of the participating
     * tensors the shortest depth is chosen to reduce the operation to a common
     * denominator (in which case the excess layers of the longer tensors are
     * left untouched).
     * 
     * @param alpha
     *            scalar scale factor for the multiplication
     * @param B
     *            tensor to be multiplied from the right
     * @param C
     *            the tensor to add on input, contains the result of the
     *            operation on output
     * @return {@code C}
     */
    public TensorF transAmultAdd(float alpha, TensorF B, TensorF C) {
        Checks.checkTransAmultAdd(this, B, C);
        int _depth = Math.min(Math.min(this.depth, B.depth), C.depth);
        Blas blas = Matrices.getBlas();
        blas.sgemm_multi(TTrans.TRANS.val(), TTrans.NO_TRANS.val(), C.numRows(), C.numColumns(), rows, alpha, a, OFFS,
                Math.max(1, rows), B.getArrayUnsafe(), OFFS, Math.max(1, B.numRows()), BETA, C.getArrayUnsafe(), OFFS,
                Math.max(1, C.numRows()), _depth, this.stride(), B.stride(), C.stride());
        return C;
    }

    /**
     * <code>C = alpha * A * B<sup>T</sup> + C</code> where {@code A} is this
     * tensor. On exit, the tensor {@code C} is overwritten by the result of the
     * operation. If there is a mismatch between the depths of the participating
     * tensors the shortest depth is chosen to reduce the operation to a common
     * denominator (in which case the excess layers of the longer tensors are
     * left untouched).
     * 
     * @param alpha
     *            scalar scale factor for the multiplication
     * @param B
     *            tensor whose transpose is to be multiplied from the right
     * @param C
     *            the tensor to add on input, contains the result of the
     *            operation on output
     * @return {@code C}
     */
    public TensorF transBmultAdd(float alpha, TensorF B, TensorF C) {
        Checks.checkTransBmultAdd(this, B, C);
        int _depth = Math.min(Math.min(this.depth, B.depth), C.depth);
        Blas blas = Matrices.getBlas();
        blas.sgemm_multi(TTrans.NO_TRANS.val(), TTrans.TRANS.val(), C.numRows(), C.numColumns(), cols, alpha, a, OFFS,
                Math.max(1, rows), B.getArrayUnsafe(), OFFS, Math.max(1, B.numRows()), BETA, C.getArrayUnsafe(), OFFS,
                Math.max(1, C.numRows()), _depth, this.stride(), B.stride(), C.stride());
        return C;
    }

    /**
     * <code>C = alpha * A<sup>T</sup> * B<sup>T</sup> + C</code> where
     * {@code A} is this tensor. On exit, the tensor {@code C} is overwritten by
     * the result of the operation. If there is a mismatch between the depths of
     * the participating tensors the shortest depth is chosen to reduce the
     * operation to a common denominator (in which case the excess layers of the
     * longer tensors are left untouched).
     * 
     * @param alpha
     *            scalar scale factor for the multiplication
     * @param B
     *            tensor whose transpose is to be multiplied from the right
     * @param C
     *            the tensor to add on input, contains the result of the
     *            operation on output
     * @return {@code C}
     */
    public TensorF transABmultAdd(float alpha, TensorF B, TensorF C) {
        Checks.checkTransABmultAdd(this, B, C);
        int _depth = Math.min(Math.min(this.depth, B.depth), C.depth);
        Blas blas = Matrices.getBlas();
        blas.sgemm_multi(TTrans.TRANS.val(), TTrans.TRANS.val(), C.numRows(), C.numColumns(), rows, alpha, a, OFFS,
                Math.max(1, rows), B.getArrayUnsafe(), OFFS, Math.max(1, B.numRows()), BETA, C.getArrayUnsafe(), OFFS,
                Math.max(1, C.numRows()), _depth, this.stride(), B.stride(), C.stride());
        return C;
    }

    /**
     * <code>C = A<sup>T</sup> * B<sup>T</sup> + C</code> where {@code A} is
     * this tensor. On exit, the tensor {@code C} is overwritten by the result
     * of the operation. If there is a mismatch between the depths of the
     * participating tensors the shortest depth is chosen to reduce the
     * operation to a common denominator (in which case the excess layers of the
     * longer tensors are left untouched).
     * 
     * @param B
     *            tensor whose transpose is to be multiplied from the right
     * @param C
     *            the tensor to add on input, contains the result of the
     *            operation on output
     * @return {@code C}
     */
    public TensorF transABmultAdd(TensorF B, TensorF C) {
        return transABmultAdd(1.0f, B, C);
    }

    /**
     * <code>C = alpha * A<sup>T</sup> * B<sup>T</sup></code> where {@code A} is
     * this tensor. On exit, the tensor {@code C} is overwritten by the result
     * of the operation. If there is a mismatch between the depths of the
     * participating tensors the shortest depth is chosen to reduce the
     * operation to a common denominator (in which case the excess layers of the
     * longer tensors are left untouched).
     * 
     * @param alpha
     *            scalar scale factor for the multiplication
     * @param B
     *            tensor whose transpose is to be multiplied from the right
     * @param C
     *            output tensor for the result of the multiplication
     * @return {@code C}
     */
    public TensorF transABmult(float alpha, TensorF B, TensorF C) {
        return transABmultAdd(alpha, B, C.zeroInplace());
    }

    /**
     * <code>C = A<sup>T</sup> * B<sup>T</sup></code> where {@code A} is this
     * tensor. On exit, the tensor {@code C} is overwritten by the result of the
     * operation. If there is a mismatch between the depths of the participating
     * tensors the shortest depth is chosen to reduce the operation to a common
     * denominator (in which case the excess layers of the longer tensors are
     * left untouched).
     * 
     * @param B
     *            tensor whose transpose is to be multiplied from the right
     * @param C
     *            output tensor for the result of the multiplication
     * @return {@code C}
     */
    public TensorF transABmult(TensorF B, TensorF C) {
        return transABmult(1.0f, B, C);
    }

    /**
     * <code>C = A * B<sup>T</sup> + C</code> where {@code A} is this tensor. On
     * exit, the tensor {@code C} is overwritten by the result of the operation.
     * If there is a mismatch between the depths of the participating tensors
     * the shortest depth is chosen to reduce the operation to a common
     * denominator (in which case the excess layers of the longer tensors are
     * left untouched).
     * 
     * @param B
     *            tensor whose transpose is to be multiplied from the right
     * @param C
     *            the tensor to add on input, contains the result of the
     *            operation on output
     * @return {@code C}
     */
    public TensorF transBmultAdd(TensorF B, TensorF C) {
        return transBmultAdd(1.0f, B, C);
    }

    /**
     * <code>C = alpha * A * B<sup>T</sup></code> where {@code A} is this
     * tensor. On exit, the tensor {@code C} is overwritten by the result of the
     * operation. If there is a mismatch between the depths of the participating
     * tensors the shortest depth is chosen to reduce the operation to a common
     * denominator (in which case the excess layers of the longer tensors are
     * left untouched).
     * 
     * @param alpha
     *            scalar scale factor for the multiplication
     * @param B
     *            tensor whose transpose is to be multiplied from the right
     * @param C
     *            output tensor for the result of the multiplication
     * @return {@code C}
     */
    public TensorF transBmult(float alpha, TensorF B, TensorF C) {
        return transBmultAdd(alpha, B, C.zeroInplace());
    }

    /**
     * <code>C = A * B<sup>T</sup></code> where {@code A} is this tensor. On
     * exit, the tensor {@code C} is overwritten by the result of the operation.
     * If there is a mismatch between the depths of the participating tensors
     * the shortest depth is chosen to reduce the operation to a common
     * denominator (in which case the excess layers of the longer tensors are
     * left untouched).
     * 
     * @param B
     *            tensor whose transpose is to be multiplied from the right
     * @param C
     *            output tensor for the result of the multiplication
     * @return {@code C}
     */
    public TensorF transBmult(TensorF B, TensorF C) {
        return transBmult(1.0f, B, C);
    }

    /**
     * <code>C = A<sup>T</sup> * B + C</code> where {@code A} is this tensor. On
     * exit, the tensor {@code C} is overwritten by the result of the operation.
     * If there is a mismatch between the depths of the participating tensors
     * the shortest depth is chosen to reduce the operation to a common
     * denominator (in which case the excess layers of the longer tensors are
     * left untouched).
     * 
     * @param B
     *            tensor to be multiplied from the right
     * @param C
     *            the tensor to add on input, contains the result of the
     *            operation on output
     * @return {@code C}
     */
    public TensorF transAmultAdd(TensorF B, TensorF C) {
        return transAmultAdd(1.0f, B, C);
    }

    /**
     * <code>C = alpha * A<sup>T</sup> * B</code> where {@code A} is this
     * tensor. On exit, the tensor {@code C} is overwritten by the result of the
     * operation. If there is a mismatch between the depths of the participating
     * tensors the shortest depth is chosen to reduce the operation to a common
     * denominator (in which case the excess layers of the longer tensors are
     * left untouched).
     * 
     * @param alpha
     *            scalar scale factor for the multiplication
     * @param B
     *            tensor to be multiplied from the right
     * @param C
     *            output tensor for the result of the multiplication
     * @return {@code C}
     */
    public TensorF transAmult(float alpha, TensorF B, TensorF C) {
        return transAmultAdd(alpha, B, C.zeroInplace());
    }

    /**
     * <code>C = A<sup>T</sup> * B</code> where {@code A} is this tensor. On
     * exit, the tensor {@code C} is overwritten by the result of the operation.
     * If there is a mismatch between the depths of the participating tensors
     * the shortest depth is chosen to reduce the operation to a common
     * denominator (in which case the excess layers of the longer tensors are
     * left untouched).
     * 
     * @param B
     *            tensor to be multiplied from the right
     * @param C
     *            output tensor for the result of the multiplication
     * @return {@code C}
     */
    public TensorF transAmult(TensorF B, TensorF C) {
        return transAmult(1.0f, B, C);
    }

    /**
     * {@code C = A * B + C} where {@code A} is this tensor. On exit, the tensor
     * {@code C} is overwritten by the result of the operation. If there is a
     * mismatch between the depths of the participating tensors the shortest
     * depth is chosen to reduce the operation to a common denominator (in which
     * case the excess layers of the longer tensors are left untouched).
     * 
     * @param B
     *            tensor to be multiplied from the right
     * @param C
     *            the tensor to add on input, contains the result of the
     *            operation on output
     * @return {@code C}
     */
    public TensorF multAdd(TensorF B, TensorF C) {
        return multAdd(1.0f, B, C);
    }

    /**
     * {@code C = alpha * A * B} where {@code A} is this tensor. If there is a
     * mismatch between the depths of the participating tensors the shortest
     * depth is chosen to reduce the operation to a common denominator (in which
     * case the excess layers of the longer tensors are left untouched).
     * 
     * @param alpha
     *            scalar scale factor for the multiplication
     * @param B
     *            tensor to be multiplied from the right
     * @param C
     *            output tensor for the result of the multiplication
     * @return {@code C}
     */
    public TensorF mult(float alpha, TensorF B, TensorF C) {
        return multAdd(alpha, B, C.zeroInplace());
    }

    /**
     * {@code C = A * B} where {@code A} is this tensor. If there is a mismatch
     * between the depths of the participating tensors the shortest depth is
     * chosen to reduce the operation to a common denominator (in which case the
     * excess layers of the longer tensors are left untouched).
     * 
     * @param B
     *            tensor to be multiplied from the right
     * @param C
     *            output tensor for the result of the multiplication
     * @return {@code C}
     */
    public TensorF mult(TensorF B, TensorF C) {
        return mult(1.0f, B, C);
    }

    /**
     * Hadamard product {@code C = A} &SmallCircle; {@code B} (also known as
     * element-wise product) of this tensor (A) and B. If there is a mismatch
     * between the depths of the participating tensors the shortest depth is
     * chosen to reduce the operation to a common denominator (in which case the
     * excess layers of the longer tensors are left untouched).
     * 
     * @param B
     *            the tensor this tensor is multiplied with
     * @param out
     *            output tensor for the result of the multiplication
     * @return {@code out}
     */
    public TensorF hadamard(TensorF B, TensorF out) {
        Checks.checkEqualDimension(this, B);
        Checks.checkEqualDimension(this, out);
        int _depth = Math.min(Math.min(this.depth, B.depth), out.depth);
        int _length = _depth * stride();
        float[] _a = a;
        float[] _b = B.getArrayUnsafe();
        float[] _c = out.getArrayUnsafe();
        for (int i = 0; i < _length; ++i) {
            _c[i] = _a[i] * _b[i];
        }
        return out;
    }

    /**
     * Hadamard product {@code A} &SmallCircle; {@code B} (also known as
     * element-wise product) of this tensor (A) and B. If there is a mismatch
     * between the depths of the participating tensors the shortest depth is
     * chosen to reduce the operation to a common denominator (in which case the
     * excess layers of the longer tensor are left untouched).
     * 
     * @param B
     *            the tensor this tensor is multiplied with
     * @return the result of the Hadamard multiplication
     */
    public TensorF hadamard(TensorF B) {
        Checks.checkEqualDimension(this, B);
        return hadamard(B, create(rows, cols, Math.min(this.depth, B.depth)));
    }

    /**
     * <code>A</code> &SmallCircle; <code>B<sup>T</sup></code> Hadamard
     * multiplication (also known as element-wise product) between this tensor
     * ({@code A}) and the transpose of {@code B} (<code>B<sup>T</sup></code>).
     * If there is a mismatch between the depths of the participating tensors
     * the shortest depth is chosen to reduce the operation to a common
     * denominator (in which case the excess layers of the longer tensor are
     * left untouched).
     * 
     * @param B
     *            the tensor whose transpose is multiplied with this tensor
     * @return the result of the Hadamard multiplication
     */
    public TensorF hadamardTransposed(TensorF B) {
        Checks.checkTrans(this, B);
        int _rows = rows;
        int _cols = cols;
        int _depth = Math.min(this.depth, B.depth);
        TensorF C = create(_rows, _cols, _depth);
        float[] _a = a;
        float[] _b = B.getArrayUnsafe();
        float[] _c = C.getArrayUnsafe();
        TensorBase btb = (TensorBase) B;
        for (int layer = 0; layer < _depth; ++layer) {
            for (int col = 0; col < _cols; ++col) {
                for (int row = 0; row < _rows; ++row) {
                    int idx = idx(row, col, layer);
                    _c[idx] = _a[idx] * _b[btb.idx(col, row, layer)];
                }
            }
        }
        return C;
    }

    /**
     * <code>A<sup>T</sup></code> &SmallCircle; <code>B</code> Hadamard
     * multiplication (also known as element-wise product) between this tensor
     * ({@code A}) transposed (<code>A<sup>T</sup></code>) and {@code B}. If
     * there is a mismatch between the depths of the participating tensors the
     * shortest depth is chosen to reduce the operation to a common denominator
     * (in which case the excess layers of the longer tensor are left
     * untouched).
     * 
     * @param B
     *            the tensor which is multiplied with this tensor transposed
     * @return the result of the Hadamard multiplication
     */
    public TensorF transposedHadamard(TensorF B) {
        Checks.checkTrans(this, B);
        int _rows = B.numRows();
        int _cols = B.numColumns();
        int _depth = Math.min(this.depth, B.depth);
        TensorF C = create(_rows, _cols, _depth);
        float[] _a = a;
        float[] _b = B.getArrayUnsafe();
        float[] _c = C.getArrayUnsafe();
        TensorBase btb = (TensorBase) B;
        for (int layer = 0; layer < _depth; ++layer) {
            for (int col = 0; col < _cols; ++col) {
                for (int row = 0; row < _rows; ++row) {
                    int idx = btb.idx(row, col, layer);
                    _c[idx] = _b[idx] * _a[idx(col, row, layer)];
                }
            }
        }
        return C;
    }

    /**
     * {@code A * B} convenience multiplication. None of the operands are
     * mutated. If there is a mismatch between the depths of the participating
     * tensors the shortest depth is chosen to reduce the operation to a common
     * denominator (in which case the excess layers of the longer tensor are
     * left untouched).
     * 
     * @param B
     *            second multiplicand
     * @return the result of the multiplication
     */
    public TensorF times(TensorF B) {
        return mult(B, create(this.rows, B.numColumns(), Math.min(this.depth, B.depth)));
    }

    /**
     * <code>A * B<sup>T</sup></code> multiplication. None of the operands are
     * mutated. If there is a mismatch between the depths of the participating
     * tensors the shortest depth is chosen to reduce the operation to a common
     * denominator (in which case the excess layers of the longer tensor are
     * left untouched). For the reversed order multiplication
     * <code>A<sup>T</sup> * B</code> use {@link #transposedTimes(TensorF)}.
     * 
     * @param B
     *            second multiplicand
     * @return the result of the multiplication
     */
    public TensorF timesTransposed(TensorF B) {
        return transBmult(B, create(this.rows, B.numRows(), Math.min(this.depth, B.depth)));
    }

    /**
     * <code>A<sup>T</sup> * B</code> multiplication. None of the operands are
     * mutated. If there is a mismatch between the depths of the participating
     * tensors the shortest depth is chosen to reduce the operation to a common
     * denominator (in which case the excess layers of the longer tensor are
     * left untouched). For the reversed order multiplication
     * <code>A * B<sup>T</sup></code> use {@link #timesTransposed(TensorF)}.
     * 
     * @param B
     *            second multiplicand
     * @return the result of the multiplication
     */
    public TensorF transposedTimes(TensorF B) {
        return transAmult(B, create(this.cols, B.numColumns(), Math.min(this.depth, B.depth)));
    }

    /**
     * Set all elements of this tensor to {@code 0.0} mutating this tensor.
     * 
     * @return this tensor (mutated)
     */
    public TensorF zeroInplace() {
        Arrays.fill(a, 0.0f);
        return this;
    }

    /**
     * {@code A = alpha * A}
     * 
     * @param alpha
     *            the scaling factor to apply
     * @return this tensor (mutated)
     */
    public TensorF scaleInplace(float alpha) {
        if (alpha == 0.0) {
            return zeroInplace();
        }
        if (alpha == 1.0) {
            return this;
        }
        float[] _a = a;
        for (int i = 0; i < _a.length; ++i) {
            _a[i] *= alpha;
        }
        return this;
    }

    /**
     * Clamps all elements in this tensor into the range {@code [min, max]}.
     * 
     * @param min
     *            lower-bound of the range to be clamped to
     * @param max
     *            upper-bound of the range to be clamped to
     * @return this tensor clamped in-place
     */
    public TensorF clampInplace(float min, float max) {
        float[] _a = a;
        for (int i = 0; i < _a.length; ++i) {
            _a[i] = Math.min(Math.max(_a[i], min), max);
        }
        return this;
    }

    /**
     * Get the reference to the internal backing array without copying.
     * 
     * @return the reference to the internal backing array
     */
    public float[] getArrayUnsafe() {
        return a;
    }

    /**
     * Create a copy of this tensor. The underlying storage gets copied.
     * 
     * @return a copy of this tensor
     */
    public TensorF copy() {
        return new TensorF(this);
    }

    private TensorF create(int rows, int cols, int depth) {
        return new TensorF(rows, cols, depth);
    }

    private float[] growAndCopyForAppend(Dimensions B) {
        return copyForAppend(new float[checkNewArrayLength(B)]);
    }

//    private float[] growAndCopyForAppend(int rows, int cols) {
//        return copyForAppend(new float[checkNewArrayLength(rows, cols)]);
//    }

    private float[] copyForAppend(float[] newArray) {
        System.arraycopy(a, 0, newArray, 0, length);
        return newArray;
    }
}
