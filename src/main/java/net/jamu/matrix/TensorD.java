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

/**
 * A {@code TensorD} is a 3-dimensional stack of 2-dimensional dense matrices of
 * primitive doubles with column-major storage layout. A Tensor of
 * {@code depth = 1} is a single matrix. The addressing is zero based. All
 * operations throw a {@code NullPointerException} if any of the method
 * arguments is {@code null}.
 * 
 * @since 1.4.0
 */
public class TensorD extends TensorBase {

    protected double[] a;

    /**
     * TODO
     * 
     * @param rows
     * @param cols
     */
    public TensorD(int rows, int cols) {
        this(rows, cols, 1);
    }

    /**
     * TODO
     * 
     * @param rows
     * @param cols
     * @param depth
     */
    public TensorD(int rows, int cols, int depth) {
        super(rows, cols, depth);
        a = new double[length];
    }

    /**
     * TODO
     * 
     * @param B
     * @return
     */
    public TensorD append(MatrixD B) {
        Checks.checkEqualDimension(this, B);
        double[] tmp = growAndCopyForAppend(B);
        double[] _b = B.getArrayUnsafe();
        System.arraycopy(_b, 0, tmp, length, _b.length);
        a = tmp;
        length = tmp.length;
        return this;
    }

    private double[] growAndCopyForAppend(Dimensions B) {
        return copyForAppend(new double[checkNewArrayLength(B)]);
    }

    private double[] growAndCopyForAppend(int rows, int cols) {
        return copyForAppend(new double[checkNewArrayLength(rows, cols)]);
    }

    private double[] copyForAppend(double[] newArray) {
        System.arraycopy(a, 0, newArray, 0, length);
        return newArray;
    }
}
