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
        // XXX
        return null;
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
     * Set all elements of this tensor to {@code 0.0} mutating this tensor.
     * 
     * @return this tensor (mutated)
     */
    public TensorF zeroInplace() {
        Arrays.fill(a, 0.0f);
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

    private float[] growAndCopyForAppend(Dimensions B) {
        return copyForAppend(new float[checkNewArrayLength(B)]);
    }

    private float[] growAndCopyForAppend(int rows, int cols) {
        return copyForAppend(new float[checkNewArrayLength(rows, cols)]);
    }

    private float[] copyForAppend(float[] newArray) {
        System.arraycopy(a, 0, newArray, 0, length);
        return newArray;
    }
}
