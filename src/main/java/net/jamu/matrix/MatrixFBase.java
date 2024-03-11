/*
 * Copyright 2019, 2024 Stefan Zobel
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

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A {@code MatrixFBase} is a partial implementation of a dense matrix of
 * primitive floats with column-major storage layout. The addressing is zero
 * based. All operations throw a {@code NullPointerException} if any of the
 * method arguments is {@code null}.
 */
public abstract class MatrixFBase extends DimensionsBase implements MatrixF {

    protected final float[] a;

    /**
     * Create a new {@code MatrixFBase} of dimension {@code (rows, cols)} with
     * its matrix elements set to the content of {@code array} in column-major
     * order. If {@code doArrayCopy} is {@code false} the passed {@code array}
     * is referenced directly, otherwise it is copied into a newly allocated
     * internal array.
     * 
     * @param rows
     *            number of matrix rows
     * @param cols
     *            number of matrix columns
     * @param array
     *            float array that determines the elements in this matrix in
     *            column-major order
     * @param doArrayCopy
     *            if {@code true} the array is copied, otherwise it is
     *            referenced directly
     */
    public MatrixFBase(int rows, int cols, float[] array, boolean doArrayCopy) {
        super(rows, cols, false, Float.TYPE);
        checkArrayLength(array, rows, cols);
        if (doArrayCopy) {
            float[] copy = new float[array.length];
            System.arraycopy(array, 0, copy, 0, copy.length);
            a = copy;
        } else {
            a = array;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float toScalar() {
        if (!isScalar()) {
            throw new IllegalStateException("(" + rows + " x " + cols + ") matrix is not a scalar");
        }
        return a[0];
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF scaleInplace(float alpha) {
        if (alpha == 0.0f) {
            return this.zeroInplace();
        }
        if (alpha == 1.0f) {
            return this;
        }
        float[] _a = a;
        for (int i = 0; i < _a.length; ++i) {
            _a[i] *= alpha;
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF scale(float alpha, MatrixF B) {
        Checks.checkEqualDimension(this, B);
        if (alpha == 0.0f) {
            Arrays.fill(B.getArrayUnsafe(), 0.0f);
            return B;
        }
        float[] _a = a;
        float[] _b = B.getArrayUnsafe();
        for (int i = 0; i < _b.length; ++i) {
            _b[i] = alpha * _a[i];
        }
        return B;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF trans(MatrixF AT) {
        Checks.checkTrans(this, AT);
        int cols_ = cols;
        int rows_ = rows;
        for (int col = 0; col < cols_; ++col) {
            for (int row = 0; row < rows_; ++row) {
                AT.setUnsafe(col, row, getUnsafe(row, col));
            }
        }
        return AT;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF addInplace(MatrixF B) {
        return addInplace(1.0f, B);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF addInplace(float alpha, MatrixF B) {
        Checks.checkEqualDimension(this, B);
        if (alpha != 0.0f) {
            float[] _a = a;
            float[] _b = B.getArrayUnsafe();
            for (int i = 0; i < _b.length; ++i) {
                _a[i] += alpha * _b[i];
            }
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF add(MatrixF B, MatrixF C) {
        return add(1.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF add(float alpha, MatrixF B, MatrixF C) {
        Checks.checkAdd(this, B, C);
        if (alpha == 0.0f) {
            System.arraycopy(a, 0, C.getArrayUnsafe(), 0, a.length);
        } else {
            float[] _a = a;
            float[] _b = B.getArrayUnsafe();
            float[] _c = C.getArrayUnsafe();
            for (int i = 0; i < _a.length; ++i) {
                _c[i] = _a[i] + alpha * _b[i];
            }
        }
        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF addBroadcastedVectorInplace(MatrixF B) {
        Checks.checkSameRows(this, B);
        if (this.numColumns() == B.numColumns()) {
            return addInplace(B);
        }
        if (B.numColumns() == 1) {
            float[] _a = a;
            float[] _b = B.getArrayUnsafe();
            int cols_ = cols;
            int rows_ = rows;
            for (int col = 0; col < cols_; ++col) {
                for (int row = 0; row < rows_; ++row) {
                    _a[idx(row, col)] += _b[row];
                }
            }
            return this;
        }
        // incompatible dimensions
        throw Checks.getSameColsException(this, B);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF mult(MatrixF B, MatrixF C) {
        return mult(1.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF mult(float alpha, MatrixF B, MatrixF C) {
        return multAdd(alpha, B, C.zeroInplace());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF multAdd(MatrixF B, MatrixF C) {
        return multAdd(1.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract MatrixF multAdd(float alpha, MatrixF B, MatrixF C);

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transABmult(MatrixF B, MatrixF C) {
        return transABmult(1.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transABmult(float alpha, MatrixF B, MatrixF C) {
        return transABmultAdd(alpha, B, C.zeroInplace());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transAmult(MatrixF B, MatrixF C) {
        return transAmult(1.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transAmult(float alpha, MatrixF B, MatrixF C) {
        return transAmultAdd(alpha, B, C.zeroInplace());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transBmult(MatrixF B, MatrixF C) {
        return transBmult(1.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transBmult(float alpha, MatrixF B, MatrixF C) {
        return transBmultAdd(alpha, B, C.zeroInplace());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transABmultAdd(MatrixF B, MatrixF C) {
        return transABmultAdd(1.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract MatrixF transABmultAdd(float alpha, MatrixF B, MatrixF C);

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transAmultAdd(MatrixF B, MatrixF C) {
        return transAmultAdd(1.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract MatrixF transAmultAdd(float alpha, MatrixF B, MatrixF C);

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transBmultAdd(MatrixF B, MatrixF C) {
        return transBmultAdd(1.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract MatrixF transBmultAdd(float alpha, MatrixF B, MatrixF C);

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF zeroInplace() {
        Arrays.fill(a, 0.0f);
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF setInplace(MatrixF other) {
        Checks.checkEqualDimension(this, other);
        float[] _a = a;
        float[] _b = other.getArrayUnsafe();
        System.arraycopy(_b, 0, _a, 0, _a.length);
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF setInplace(float alpha, MatrixF other) {
        Checks.checkEqualDimension(this, other);
        if (alpha == 0.0f) {
            return zeroInplace();
        }
        if (other == this) {
            return scaleInplace(alpha);
        }
        float[] _a = a;
        float[] _b = other.getArrayUnsafe();
        for (int i = 0; i < _b.length; ++i) {
            _a[i] = alpha * _b[i];
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF setInplaceUpperTrapezoidal(MatrixF B) {
        Checks.checkB_hasAtLeastAsManyColsAsA(this, B);
        Checks.checkB_hasAtLeastAsManyRowsAsARowColMin(this, B);
        int cols_ = cols;
        int rows_ = rows;
        for (int col = 0; col < cols_; ++col) {
            for (int row = 0; row < rows_; ++row) {
                if (row <= col) {
                    this.setUnsafe(row, col, B.getUnsafe(row, col));
                } else {
                    this.setUnsafe(row, col, 0.0f);
                }
            }
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF setInplaceLowerTrapezoidal(MatrixF B) {
        Checks.checkB_hasAtLeastAsManyRowsAsA(this, B);
        Checks.checkB_hasAtLeastAsManyColsAsARowColMin(this, B);
        int cols_ = cols;
        int rows_ = rows;
        for (int col = 0; col < cols_; ++col) {
            for (int row = 0; row < rows_; ++row) {
                if (row >= col) {
                    this.setUnsafe(row, col, B.getUnsafe(row, col));
                } else {
                    this.setUnsafe(row, col, 0.0f);
                }
            }
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF submatrix(int r0, int c0, int r1, int c1, MatrixF B, int rb, int cb) {
        checkSubmatrixIndexes(r0, c0, r1, c1);
        B.checkIndex(rb, cb);
        B.checkIndex(rb + r1 - r0, cb + c1 - c0);
        int rbStart = rb;
        for (int col = c0; col <= c1; ++col) {
            for (int row = r0; row <= r1; ++row) {
                B.setUnsafe(rb++, cb, this.getUnsafe(row, col));
            }
            rb = rbStart;
            cb++;
        }
        return B;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF setSubmatrixInplace(int r0, int c0, MatrixF B, int rb0, int cb0, int rb1, int cb1) {
        B.checkSubmatrixIndexes(rb0, cb0, rb1, cb1);
        checkIndex(r0, c0);
        checkIndex(r0 + rb1 - rb0, c0 + cb1 - cb0);
        int r0Start = r0;
        for (int col = cb0; col <= cb1; ++col) {
            for (int row = rb0; row <= rb1; ++row) {
                this.setUnsafe(r0++, c0, B.getUnsafe(row, col));
            }
            r0 = r0Start;
            c0++;
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float get(int row, int col) {
        checkIndex(row, col);
        return a[idx(row, col)];
    }

    /**
     * {@inheritDoc}
     */
    public float getUnsafe(int row, int col) {
        return a[idx(row, col)];
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF set(int row, int col, float val) {
        checkIndex(row, col);
        a[idx(row, col)] = val;
        return this;
    }

    /**
     * {@inheritDoc}
     */
    public void setUnsafe(int row, int col, float val) {
        a[idx(row, col)] = val;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF add(int row, int col, float val) {
        checkIndex(row, col);
        a[idx(row, col)] += val;
        return this;
    }

    protected void addUnsafe(int row, int col, float val) {
        a[idx(row, col)] += val;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float[] getArrayUnsafe() {
        return a;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float[][] toJaggedArray() {
        int _rows = rows;
        int _cols = cols;
        float[] _a = a;
        float[][] copy = new float[_rows][_cols];
        for (int row = 0; row < _rows; ++row) {
            float[] row_i = copy[row];
            for (int col = 0; col < row_i.length; ++col) {
                row_i[col] = _a[col * _rows + row];
            }
        }
        return copy;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF inv(MatrixF inverse) {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("The inverse is only defined for square matrices");
        }
        Checks.checkEqualDimension(this, inverse);
        return solve(Matrices.identityF(this.numRows()), inverse);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF pseudoInv() {
        if (this.isSquareMatrix()) {
            return inv(create(rows, cols));
        }
        SvdF svd = svd(true);
        float[] sigma = svd.getS();
        float tol = MACH_EPS_FLT * Math.max(rows, cols) * sigma[0];
        // Sigma dagger
        MatrixF SInv = create(cols, rows);
        for (int i = 0; i < sigma.length; ++i) {
            if (sigma[i] > tol) {
                SInv.setUnsafe(i, i, 1.0f / sigma[i]);
            }
        }
        // Vt transposed times SInv
        MatrixF Vt = svd.getVt();
        MatrixF x = Vt.transAmult(SInv, create(Vt.numRows(), SInv.numColumns()));
        // x times U transposed (the Moore-Penrose pseudoinverse)
        MatrixF U = svd.getU();
        return x.transBmult(U, create(x.numRows(), U.numRows()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF expm() {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("Matrix exponentiation is only defined for square matrices");
        }
        return Expm.expmF(this, normMaxAbs());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF hadamard(MatrixF B, MatrixF out) {
        Checks.checkEqualDimension(this, B);
        Checks.checkEqualDimension(this, out);
        float[] _a = a;
        float[] _b = B.getArrayUnsafe();
        float[] _c = out.getArrayUnsafe();
        for (int i = 0; i < _a.length; ++i) {
            _c[i] = _a[i] * _b[i];
        }
        return out;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float normF() {
        // overflow resistant implementation
        double scale = 0.0;
        double sumsquared = 1.0;
        float[] _a = a;
        for (int i = 0; i < _a.length; ++i) {
            double xi = _a[i];
            if (xi != 0.0) {
                double absxi = Math.abs(xi);
                if (scale < absxi) {
                    double unsquared = scale / absxi;
                    sumsquared = 1.0 + sumsquared * (unsquared * unsquared);
                    scale = absxi;
                } else {
                    double unsquared = absxi / scale;
                    sumsquared = sumsquared + (unsquared * unsquared);
                }
            }
        }
        return (float) (scale * Math.sqrt(sumsquared));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float normMaxAbs() {
        float max = Float.NEGATIVE_INFINITY;
        float[] _a = a;
        for (int i = 0; i < _a.length; ++i) {
            float xi = _a[i];
            if (xi > max) {
                max = xi;
            }
        }
        return max;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float normInf() {
        double max = 0.0;
        int rows_ = rows;
        int cols_ = cols;
        for (int i = 0; i < rows_; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols_; j++) {
                sum += Math.abs(getUnsafe(i, j));
            }
            max = Math.max(max, sum);
        }
        return (float) max;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float norm1() {
        double max = 0.0;
        float[] _a = a;
        int rows_ = rows;
        int cols_ = cols;
        for (int col = 0; col < cols_; ++col) {
            double sum = 0.0;
            for (int idx = col * rows_; idx < (col + 1) * rows_; ++idx) {
                sum += Math.abs(_a[idx]);
            }
            max = Math.max(max, sum);
        }
        return (float) max;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float trace() {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("The trace of a matrix is only defined for square matrices");
        }
        float t = 0.0f;
        for (int i = 0; i < rows; ++i) {
            t += getUnsafe(i, i);
        }
        return t;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF clampInplace(float min, float max) {
        float[] _a = a;
        for (int i = 0; i < _a.length; ++i) {
            _a[i] = Math.min(Math.max(_a[i], min), max);
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF mapInplace(FFunction f) {
        float[] _a = a;
        for (int i = 0; i < _a.length; ++i) {
            _a[i] = f.apply(_a[i]);
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF zeroizeSubEpsilonInplace(int k) {
        if (k < 1) {
            throw new IllegalArgumentException("Illegal multiplier < 1 : " + k);
        }
        float threshold = k * MACH_EPS_FLT;
        float[] _a = a;
        for (int i = 0; i < _a.length; ++i) {
            if (Math.abs(_a[i]) <= threshold) {
                _a[i] = 0.0f;
            }
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF sanitizeNonFiniteInplace(float nanSurrogate, float posInfSurrogate, float negInfSurrogate) {
        boolean subNan = (nanSurrogate == nanSurrogate); // "lgtm[java/comparison-of-identical-expressions]"
        boolean subPInf = (posInfSurrogate != Float.POSITIVE_INFINITY);
        boolean subNInf = (negInfSurrogate != Float.NEGATIVE_INFINITY);
        if (!subNan && !subPInf && !subNInf) {
            return this;
        }
        float[] _a = a;
        for (int i = 0; i < _a.length; ++i) {
            float x = _a[i];
            if (x != x && subNan) { // "lgtm[java/comparison-of-identical-expressions]"
                _a[i] = nanSurrogate;
            } else if (x == Float.POSITIVE_INFINITY && subPInf) {
                _a[i] = posInfSurrogate;
            } else if (x == Float.NEGATIVE_INFINITY && subNInf) {
                _a[i] = negInfSurrogate;
            }
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF sanitizeNaNInplace(float nanSurrogate) {
        return sanitizeNonFiniteInplace(nanSurrogate, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
    }

    /**
     * Returns a string representation of this matrix. If the matrix has more
     * than 6 rows and/or more than 6 columns only the first 5 contiguous rows
     * and/or columns are displayed followed by a {@code "......"} marker and
     * then the last row and/or column is displayed as the 6th row / column.
     * 
     * @return a (possibly truncated) string representation of this matrix
     */
    @Override
    public String toString() {
        return Matrices.toString(this);
    }

    // FMatrixBasicOps

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF selectColumn(int col) {
        return selectConsecutiveColumns(col, col);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF selectConsecutiveColumns(int colFrom, int colTo) {
        checkSubmatrixIndexes(0, colFrom, rows - 1, colTo);
        int startPos = rows * colFrom;
        int length = ((colTo - colFrom) + 1) * rows;
        float[] dest = new float[length];
        System.arraycopy(a, startPos, dest, 0, length);
        return create(rows, (colTo - colFrom) + 1, dest);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF selectSubmatrix(int rowFrom, int colFrom, int rowTo, int colTo) {
        checkSubmatrixIndexes(rowFrom, colFrom, rowTo, colTo);
        MatrixF copy = create(rowTo - rowFrom + 1, colTo - colFrom + 1);
        return submatrix(rowFrom, colFrom, rowTo, colTo, copy, 0, 0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF appendColumn(MatrixF colVector) {
        Checks.checkCommensurateColVector(this, colVector);
        float[] _a = a;
        float[] _b = colVector.getArrayUnsafe();
        float[] _ab = new float[rows * (cols + 1)];
        System.arraycopy(_a, 0, _ab, 0, _a.length);
        System.arraycopy(_b, 0, _ab, _a.length, _b.length);
        return create(rows, cols + 1, _ab);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF appendMatrix(MatrixF matrix) {
        Checks.checkSameRows(this, matrix);
        int colsNew = cols + matrix.numColumns();
        float[] _a = a;
        float[] _b = matrix.getArrayUnsafe();
        float[] _ab = new float[rows * colsNew];
        System.arraycopy(_a, 0, _ab, 0, _a.length);
        System.arraycopy(_b, 0, _ab, _a.length, _b.length);
        return create(rows, colsNew, _ab);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF mldivide(MatrixF B) {
        Checks.checkSameRows(this, B);
        return solve(B, Matrices.createF(cols, B.numColumns()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF mrdivide(MatrixF B) {
        Checks.checkSameCols(this, B);
        return B.transpose().mldivide(this.transpose()).transpose();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF times(MatrixF B) {
        return mult(B, create(this.rows, B.numColumns()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF timesTimes(MatrixF B, MatrixF C) {
        if (Matrices.aTimesBfirst(this, B, C)) {
            return mult(B, create(rows, B.numColumns())).mult(C, create(rows, C.numColumns()));
        } else {
            // since 1.2.1
            return mult(B.mult(C, create(B.numRows(), C.numColumns())), create(rows, C.numColumns()));
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF timesMany(MatrixF m, MatrixF... matrices) {
        int len = matrices.length;
        if (len == 0) {
            return times(m);
        }
        if (len == 1) {
            return timesTimes(m, matrices[0]);
        }
        Checks.checkMultMany(this, m, matrices);
        UPInts splits = new MatrixChain(this, m, matrices).computeSplits();
        ArrayList<MatrixF> chain = MatrixChain.buildList(this, m, matrices);
        return multiplyMany(0, chain.size() - 1, chain, splits);
    }

    /*
     * Recursive function for the actual matrix-chain multiplication, also from
     * Cormen et al.
     */
    private MatrixF multiplyMany(int i, int j, ArrayList<MatrixF> chain, UPInts splits) {
        if (i == j) {
            // base case
            return chain.get(i);
        }
        int k = splits.get(i, j);
        MatrixF X = multiplyMany(i, k, chain, splits);
        MatrixF Y = multiplyMany(k + 1, j, chain, splits);
        return X.mult(Y, create(X.numRows(), Y.numColumns()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF timesTransposed() {
        return transBmult(this, create(rows, rows));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF timesTransposed(MatrixF B) {
        return transBmult(B, create(rows, B.numRows()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transposedTimes() {
        return transAmult(this, create(cols, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transposedTimes(MatrixF B) {
        return transAmult(B, create(cols, B.numColumns()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF hadamard(MatrixF B) {
        return hadamard(B, create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF hadamardTransposed(MatrixF B) {
        Checks.checkTrans(this, B);
        int _rows = rows;
        int _cols = cols;
        MatrixF C = create(_rows, _cols);
        float[] _a = a;
        float[] _b = B.getArrayUnsafe();
        float[] _c = C.getArrayUnsafe();
        DimensionsBase bdb = (DimensionsBase) B;
        for (int col = 0; col < _cols; ++col) {
            for (int row = 0; row < _rows; ++row) {
                int idx = idx(row, col);
                _c[idx] = _a[idx] * _b[bdb.idx(col, row)];
            }
        }
        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transposedHadamard(MatrixF B) {
        Checks.checkTrans(this, B);
        int _rows = B.numRows();
        int _cols = B.numColumns();
        MatrixF C = create(_rows, _cols);
        float[] _a = a;
        float[] _b = B.getArrayUnsafe();
        float[] _c = C.getArrayUnsafe();
        DimensionsBase bdb = (DimensionsBase) B;
        for (int col = 0; col < _cols; ++col) {
            for (int row = 0; row < _rows; ++row) {
                int idx = bdb.idx(row, col);
                _c[idx] = _b[idx] * _a[idx(col, row)];
            }
        }
        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF times(ComplexMatrixF B) {
        Checks.checkMult(this, B);
        ComplexMatrixF Ac = Matrices.convertToComplex(this);
        return Ac.times(B);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transpose() {
        if (rows == 1 || cols == 1) {
            return create(cols, rows, Arrays.copyOf(a, a.length));
        }
        return trans(create(cols, rows));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF inverse() {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("The inverse is only defined for square matrices");
        }
        return inv(create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF plus(MatrixF B) {
        return add(B, create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF timesPlus(MatrixF B, MatrixF C) {
        return multAdd(B, C.copy());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF timesMinus(MatrixF B, MatrixF C) {
        return multAdd(B, C.uminus());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF minus(MatrixF B) {
        return add(-1.0f, B, create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF uminus() {
        return scale(-1.0f, create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF abs() {
        MatrixF m = copy();
        float[] b_ = m.getArrayUnsafe();
        for (int i = 0; i < b_.length; ++i) {
            float x = b_[i]; 
            if (x < 0.0f) {
                b_[i] = -x;
            }
        }
        return m;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF map(FFunction f) {
        return copy().mapInplace(f);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF reshape(int rows, int cols) {
        Checks.checkCompatibleDimension(this, rows, cols);
        return create(rows, cols, Arrays.copyOf(a, a.length));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float[] singularValues() {
        return svd(false).getS();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF toComplexMatrix() {
        return Matrices.convertToComplex(this);
    }

    // protected methods

    protected abstract MatrixF create(int rows, int cols);

    protected abstract MatrixF create(int rows, int cols, float[] data);

    protected static void checkArrayLength(float[] array, int rows, int cols) {
        if (array.length != rows * cols) {
            throw new IllegalArgumentException(
                    "data array has wrong length. Needed : " + rows * cols + " , Is : " + array.length);
        }
    }
}
