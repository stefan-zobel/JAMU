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
 * Abstract base implementation of the {@code TensorDimensions} interface. Note
 * that all addressing is zero based and that the numbers of rows and columns
 * and the depth must be strictly positive.
 * 
 * @since 1.4.0
 */
public abstract class TensorBase implements TensorDimensions {

    protected final int rows;
    protected final int cols;
    protected int depth;
    protected int length;

    /**
     * Constructs a new {@link TensorDimensions} implementation which checks
     * that the {@code rows}, {@code cols} and {@code depth} dimension
     * parameters are strictly positive.
     * 
     * @param rows
     *            number of matrix rows
     * @param cols
     *            number of matrix columns
     * @param depth
     *            number of matrices in the tensor
     */
    public TensorBase(int rows, int cols, int depth) {
        this.rows = DimensionsBase.checkRows(rows);
        this.cols = DimensionsBase.checkCols(cols);
        this.depth = checkDepth(depth);
        this.length = checkArrayLength(rows, cols, depth);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int numColumns() {
        return cols;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int numRows() {
        return rows;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int numDepth() {
        return depth;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final int stride() {
        return rows * cols;
    }

    protected int startIdx(int layer) {
        checkLayer(layer);
        return stride() * layer;
    }

    protected final int idx(int row, int col, int layer) {
        return layer * stride() + col * rows + row;
    }

    protected void checkIndex(int row, int col, int layer) {
        checkRow(row);
        checkCol(col);
        checkLayer(layer);
    }

    protected void checkRow(int row) {
        if (row < 0 || row >= rows) {
            throw new IllegalArgumentException(
                    "Illegal row index " + row + " in (" + rows + " x " + cols + " x " + depth + ") tensor");
        }
    }

    protected void checkCol(int col) {
        if (col < 0 || col >= cols) {
            throw new IllegalArgumentException(
                    "Illegal column index " + col + " in (" + rows + " x " + cols + " x " + depth + ") tensor");
        }
    }

    protected void checkLayer(int layer) {
        if (layer < 0 || layer >= depth) {
            throw new IllegalArgumentException(
                    "Illegal layer index " + layer + " in (" + rows + " x " + cols + " x " + depth + ") tensor");
        }
    }

    protected int checkNewArrayLength(Dimensions B) {
        return checkNewArrayLength(B.numRows(), B.numColumns());
    }

    protected int checkNewArrayLength(int rows, int cols) {
        long newLength = length + ((long) rows * (long) cols);
        if (newLength <= 0L || newLength >= (long) Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "A length of " + newLength + " exceeds the maximal possible length (= 2147483647) of an array");
        }
        if (newLength % stride() != 0L) {
            throw new IllegalArgumentException("A new length of " + newLength + " is not a multiple of " + stride());
        }
        return (int) newLength;
    }

    protected static int checkArrayLength(int rows, int cols, int depth) {
        long length = (long) DimensionsBase.checkRows(rows) * (long) DimensionsBase.checkCols(cols)
                * (long) checkDepth(depth);
        if (length > (long) Integer.MAX_VALUE) {
            throw new IllegalArgumentException("rows x cols x depth (= " + length
                    + ") exceeds the maximal possible length (= 2147483647) of an array");
        }
        return (int) length;
    }

    protected static int checkDepth(int depth) {
        if (depth <= 0) {
            throw new IllegalArgumentException("the tensor depth must be strictly positive : " + depth);
        }
        return depth;
    }
}
