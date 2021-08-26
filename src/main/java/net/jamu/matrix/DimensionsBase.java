/*
 * Copyright 2019, 2021 Stefan Zobel
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

import java.util.Objects;

/**
 * Abstract base implementation of the {@code Dimensions} interface. Note that
 * all addressing is zero based and that the numbers of rows and columns must be
 * strictly positive.
 */
public abstract class DimensionsBase implements Dimensions {

    /** The IEEE 754 machine epsilon from Cephes: (2^-53) */
    protected static final double MACH_EPS_DBL = 1.11022302462515654042e-16;
    /** The IEEE 754 single precision machine epsilon: (2^-24) */
    protected static final float MACH_EPS_FLT = 5.9604644775390625e-8f;

    protected final int rows;
    protected final int cols;
    protected final boolean complex;
    protected final Class<?> type;
    private String formatString;

    /**
     * Constructs a new {@link Dimensions} implementation which checks that the
     * {@code rows} and {@code cols} dimension parameters are strictly positive.
     * 
     * @param rows
     *            number of matrix rows
     * @param cols
     *            number of matrix columns
     * @param complex
     *            {@code false} if this {@code Dimensions} implementation is a
     *            real matrix, otherwise {@code true}
     * @param type
     *            the class instance representing the primitive type (either
     *            {@link Float#TYPE} or {@link Double#TYPE})
     * @throws IllegalArgumentException
     *             if any one of the dimension parameters is not strictly
     *             positive
     * @since 1.3
     */
    public DimensionsBase(int rows, int cols, boolean complex, Class<?> type) {
        checkRows(rows);
        checkCols(cols);
        this.rows = rows;
        this.cols = cols;
        this.complex = complex;
        this.type = Objects.requireNonNull(type);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isScalar() {
        return rows == 1 && cols == 1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isColumnVector() {
        return cols == 1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isRowVector() {
        return rows == 1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isSquareMatrix() {
        return rows == cols;
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
    public void checkIndex(int row, int col) {
        if (row < 0 || row >= rows) {
            throw new IllegalArgumentException("Illegal row index " + row + " in (" + rows + " x " + cols + ") matrix");
        }
        if (col < 0 || col >= cols) {
            throw new IllegalArgumentException(
                    "Illegal column index " + col + " in (" + rows + " x " + cols + ") matrix");
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void checkSubmatrixIndexes(int rFrom, int cFrom, int rTo, int cTo) {
        checkIndex(rFrom, cFrom);
        checkIndex(rTo, cTo);
        int _rows = rTo - rFrom + 1;
        int _cols = cTo - cFrom + 1;
        if (_rows <= 0 || _cols <= 0) {
            throw new IllegalArgumentException(
                    "Illegal submatrix indices : [" + rFrom + ", " + cFrom + ", " + rTo + ", " + cTo + "]");
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int startRow() {
        return 0;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int endRow() {
        return numRows() - 1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int startCol() {
        return 0;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int endCol() {
        return numColumns() - 1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isComplex() {
        return complex;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isDoublePrecision() {
        return type == Double.TYPE;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String asString() {
        return new StringBuilder("(").append(rows).append(" x ").append(cols).append(")").toString();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String getFormatString() {
        if (formatString != null) {
            return formatString;
        }
        if (isDoublePrecision()) {
            return isComplex() ? "%.10E" : "%.12E";
        }
        return isComplex() ? "%.6E" : "%.8E";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setFormatString(String formatString) {
        this.formatString = Objects.requireNonNull(formatString);
    }

    protected int idx(int row, int col) {
        return col * rows + row;
    }

    protected static int checkRows(int rows) {
        if (rows <= 0) {
            throw new IllegalArgumentException("number of rows must be strictly positive : " + rows);
        }
        return rows;
    }

    protected static int checkCols(int cols) {
        if (cols <= 0) {
            throw new IllegalArgumentException("number of columns must be strictly positive : " + cols);
        }
        return cols;
    }
}
