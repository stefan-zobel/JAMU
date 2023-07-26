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

    protected float[] a;

    /**
     * TODO
     * 
     * @param rows
     * @param cols
     */
    public TensorF(int rows, int cols) {
        this(rows, cols, 1);
    }

    /**
     * TODO
     * 
     * @param rows
     * @param cols
     * @param depth
     */
    public TensorF(int rows, int cols, int depth) {
        super(rows, cols, depth);
        a = new float[length];
    }

    /**
     * 
     * @param A
     */
    public TensorF(MatrixF A) {
        super(A.numRows(), A.numColumns(), 1);
        a = Arrays.copyOf(A.getArrayUnsafe(), A.getArrayUnsafe().length);
    }

    public TensorF set(int row, int col, int layer, float value) {
        checkIndex(row, col, layer);
        return setUnsafe(row, col, layer, value);
    }

    public float get(int row, int col, int layer) {
        checkIndex(row, col, layer);
        return getUnsafe(row, col, layer);
    }

    public TensorF setUnsafe(int row, int col, int layer, float value) {
        a[idx(row, col, layer)] = value;
        return this;
    }

    public float getUnsafe(int row, int col, int layer) {
        return a[idx(row, col, layer)];
    }

    /**
     * TODO
     * 
     * @param B
     * @param layer
     * @return
     */
    public TensorF set(MatrixF B, int layer) {
        Checks.checkEqualDimension(this, B);
        int start = startIdx(layer);
        float[] _b = B.getArrayUnsafe();
        System.arraycopy(_b, 0, a, start, _b.length);
        return this;
    }

    /**
     * TODO
     * 
     * @param layer
     * @return
     */
    public MatrixF get(int layer) {
        int start = startIdx(layer);
        int len = stride();
        float[] _a = new float[len];
        System.arraycopy(a, start, _a, 0, len);
        return new SimpleMatrixF(rows, cols, _a);
    }

    /**
     * TODO
     * 
     * @param B
     * @return
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
