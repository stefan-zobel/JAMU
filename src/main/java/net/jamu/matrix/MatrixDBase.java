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

import java.util.Arrays;

/**
 * A {@code MatrixDBase} is a partial implementation of a dense matrix of
 * primitive doubles with column-major storage layout. The addressing is zero
 * based. All operations throw a {@code NullPointerException} if any of the
 * method arguments is {@code null}.
 */
public abstract class MatrixDBase extends DimensionsBase implements MatrixD {

    protected final double[] a;

    /**
     * Create a new {@code MatrixDBase} of dimension {@code (rows, cols)} with
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
     *            double array that determines the elements in this matrix in
     *            column-major order
     * @param doArrayCopy
     *            if {@code true} the array is copied, otherwise it is
     *            referenced directly
     */
    public MatrixDBase(int rows, int cols, double[] array, boolean doArrayCopy) {
        super(rows, cols);
        checkArrayLength(array, rows, cols);
        if (doArrayCopy) {
            double[] copy = new double[array.length];
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
    public double toScalar() {
        if (!isScalar()) {
            throw new IllegalStateException("(" + rows + " x " + cols + ") matrix is not a scalar");
        }
        return a[0];
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD scaleInplace(double alpha) {
        if (alpha == 0.0) {
            return this.zeroInplace();
        }
        if (alpha == 1.0) {
            return this;
        }
        double[] _a = a;
        for (int i = 0; i < _a.length; ++i) {
            _a[i] *= alpha;
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD scale(double alpha, MatrixD B) {
        Checks.checkEqualDimension(this, B);
        if (alpha == 0.0) {
            Arrays.fill(B.getArrayUnsafe(), 0.0);
            return B;
        }
        double[] _a = a;
        double[] _b = B.getArrayUnsafe();
        for (int i = 0; i < _b.length; ++i) {
            _b[i] = alpha * _a[i];
        }
        return B;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD trans(MatrixD AT) {
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
    public MatrixD addInplace(MatrixD B) {
        return addInplace(1.0, B);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD addInplace(double alpha, MatrixD B) {
        Checks.checkEqualDimension(this, B);
        if (alpha != 0.0) {
            double[] _a = a;
            double[] _b = B.getArrayUnsafe();
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
    public MatrixD add(MatrixD B, MatrixD C) {
        return add(1.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD add(double alpha, MatrixD B, MatrixD C) {
        Checks.checkAdd(this, B, C);
        if (alpha == 0.0) {
            System.arraycopy(a, 0, C.getArrayUnsafe(), 0, a.length);
        } else {
            double[] _a = a;
            double[] _b = B.getArrayUnsafe();
            double[] _c = C.getArrayUnsafe();
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
    public MatrixD mult(MatrixD B, MatrixD C) {
        return mult(1.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD mult(double alpha, MatrixD B, MatrixD C) {
        return multAdd(alpha, B, C.zeroInplace());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD multAdd(MatrixD B, MatrixD C) {
        return multAdd(1.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract MatrixD multAdd(double alpha, MatrixD B, MatrixD C);

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD transABmult(MatrixD B, MatrixD C) {
        return transABmult(1.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD transABmult(double alpha, MatrixD B, MatrixD C) {
        return transABmultAdd(alpha, B, C.zeroInplace());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD transAmult(MatrixD B, MatrixD C) {
        return transAmult(1.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD transAmult(double alpha, MatrixD B, MatrixD C) {
        return transAmultAdd(alpha, B, C.zeroInplace());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD transBmult(MatrixD B, MatrixD C) {
        return transBmult(1.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD transBmult(double alpha, MatrixD B, MatrixD C) {
        return transBmultAdd(alpha, B, C.zeroInplace());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD transABmultAdd(MatrixD B, MatrixD C) {
        return transABmultAdd(1.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract MatrixD transABmultAdd(double alpha, MatrixD B, MatrixD C);

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD transAmultAdd(MatrixD B, MatrixD C) {
        return transAmultAdd(1.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract MatrixD transAmultAdd(double alpha, MatrixD B, MatrixD C);

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD transBmultAdd(MatrixD B, MatrixD C) {
        return transBmultAdd(1.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract MatrixD transBmultAdd(double alpha, MatrixD B, MatrixD C);

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD zeroInplace() {
        Arrays.fill(a, 0.0);
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD setInplace(MatrixD other) {
        Checks.checkEqualDimension(this, other);
        double[] _a = a;
        double[] _b = other.getArrayUnsafe();
        System.arraycopy(_b, 0, _a, 0, _a.length);
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD setInplace(double alpha, MatrixD other) {
        Checks.checkEqualDimension(this, other);
        if (alpha == 0.0) {
            return zeroInplace();
        }
        if (other == this) {
            return scaleInplace(alpha);
        }
        double[] _a = a;
        double[] _b = other.getArrayUnsafe();
        for (int i = 0; i < _b.length; ++i) {
            _a[i] = alpha * _b[i];
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD setInplaceUpperTrapezoidal(MatrixD B) {
        Checks.checkB_hasAtLeastAsManyColsAsA(this, B);
        Checks.checkB_hasAtLeastAsManyRowsAsARowColMin(this, B);
        int cols_ = cols;
        int rows_ = rows;
        for (int col = 0; col < cols_; ++col) {
            for (int row = 0; row < rows_; ++row) {
                if (row <= col) {
                    this.setUnsafe(row, col, B.getUnsafe(row, col));
                } else {
                    this.setUnsafe(row, col, 0.0);
                }
            }
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD setInplaceLowerTrapezoidal(MatrixD B) {
        Checks.checkB_hasAtLeastAsManyRowsAsA(this, B);
        Checks.checkB_hasAtLeastAsManyColsAsARowColMin(this, B);
        int cols_ = cols;
        int rows_ = rows;
        for (int col = 0; col < cols_; ++col) {
            for (int row = 0; row < rows_; ++row) {
                if (row >= col) {
                    this.setUnsafe(row, col, B.getUnsafe(row, col));
                } else {
                    this.setUnsafe(row, col, 0.0);
                }
            }
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD submatrix(int r0, int c0, int r1, int c1, MatrixD B, int rb, int cb) {
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
    public MatrixD setSubmatrixInplace(int r0, int c0, MatrixD B, int rb0, int cb0, int rb1, int cb1) {
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
    public double get(int row, int col) {
        checkIndex(row, col);
        return a[idx(row, col)];
    }

    /**
     * {@inheritDoc}
     */
    public double getUnsafe(int row, int col) {
        return a[idx(row, col)];
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD set(int row, int col, double val) {
        checkIndex(row, col);
        a[idx(row, col)] = val;
        return this;
    }

    /**
     * {@inheritDoc}
     */
    public void setUnsafe(int row, int col, double val) {
        a[idx(row, col)] = val;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD add(int row, int col, double val) {
        checkIndex(row, col);
        a[idx(row, col)] += val;
        return this;
    }

    protected void addUnsafe(int row, int col, double val) {
        a[idx(row, col)] += val;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[] getArrayUnsafe() {
        return a;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[][] toJaggedArray() {
        int _rows = rows;
        int _cols = cols;
        double[] _a = a;
        double[][] copy = new double[_rows][_cols];
        for (int row = 0; row < _rows; ++row) {
            double[] row_i = copy[row];
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
    public MatrixD inv(MatrixD inverse) {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("The inverse is only defined for square matrices");
        }
        Checks.checkEqualDimension(this, inverse);
        return solve(Matrices.identityD(this.numRows()), inverse);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD pseudoInv() {
        if (this.isSquareMatrix()) {
            return inv(create(rows, cols));
        }
        SvdD svd = svd(true);
        double[] sigma = svd.getS();
        double tol = MACH_EPS_DBL * Math.max(rows, cols) * sigma[0];
        // Sigma dagger
        MatrixD SInv = create(cols, rows);
        for (int i = 0; i < sigma.length; ++i) {
            if (sigma[i] > tol) {
                SInv.setUnsafe(i, i, 1.0 / sigma[i]);
            }
        }
        // Vt transposed times SInv
        MatrixD Vt = svd.getVt();
        MatrixD x = Vt.transAmult(SInv, create(Vt.numRows(), SInv.numColumns()));
        // x times U transposed (the Moore-Penrose pseudoinverse)
        MatrixD U = svd.getU();
        return x.transBmult(U, create(x.numRows(), U.numRows()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD expm() {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("Matrix exponentiation is only defined for square matrices");
        }
        return Expm.expmD(this, normMaxAbs());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double normF() {
        // overflow resistant implementation
        double scale = 0.0;
        double sumsquared = 1.0;
        double[] _a = a;
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
        return scale * Math.sqrt(sumsquared);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double normMaxAbs() {
        double max = Double.NEGATIVE_INFINITY;
        double[] _a = a;
        for (int i = 0; i < _a.length; ++i) {
            double xi = _a[i];
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
    public double normInf() {
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
        return max;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double norm1() {
        double max = 0.0;
        double[] _a = a;
        int rows_ = rows;
        int cols_ = cols;
        for (int col = 0; col < cols_; ++col) {
            double sum = 0.0;
            for (int idx = col * rows_; idx < (col + 1) * rows_; ++idx) {
                sum += Math.abs(_a[idx]);
            }
            max = Math.max(max, sum);
        }
        return max;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double trace() {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("The trace of a matrix is only defined for square matrices");
        }
        double t = 0.0;
        for (int i = 0; i < rows; ++i) {
            t += getUnsafe(i, i);
        }
        return t;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD zeroizeSubEpsilonInplace(int k) {
        if (k < 1) {
            throw new IllegalArgumentException("Illegal multiplier < 1 : " + k);
        }
        double threshold = k * MACH_EPS_DBL;
        double[] _a = a;
        for (int i = 0; i < _a.length; ++i) {
            if (Math.abs(_a[i]) <= threshold) {
                _a[i] = 0.0;
            }
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD sanitizeNonFiniteInplace(double nanSurrogate, double posInfSurrogate, double negInfSurrogate) {
        boolean subNan = (nanSurrogate == nanSurrogate); // "lgtm[java/comparison-of-identical-expressions]"
        boolean subPInf = (posInfSurrogate != Double.POSITIVE_INFINITY);
        boolean subNInf = (negInfSurrogate != Double.NEGATIVE_INFINITY);
        if (!subNan && !subPInf && !subNInf) {
            return this;
        }
        double[] _a = a;
        for (int i = 0; i < _a.length; ++i) {
            double x = _a[i];
            if (x != x && subNan) { // "lgtm[java/comparison-of-identical-expressions]"
                _a[i] = nanSurrogate;
            } else if (x == Double.POSITIVE_INFINITY && subPInf) {
                _a[i] = posInfSurrogate;
            } else if (x == Double.NEGATIVE_INFINITY && subNInf) {
                _a[i] = negInfSurrogate;
            }
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD sanitizeNaNInplace(double nanSurrogate) {
        return sanitizeNonFiniteInplace(nanSurrogate, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
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

    // DMatrixBasicOps

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD selectColumn(int col) {
        return selectConsecutiveColumns(col, col);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD selectConsecutiveColumns(int colFrom, int colTo) {
        checkSubmatrixIndexes(0, colFrom, rows - 1, colTo);
        int startPos = rows * colFrom;
        int length = ((colTo - colFrom) + 1) * rows;
        double[] dest = new double[length];
        System.arraycopy(a, startPos, dest, 0, length);
        return create(rows, (colTo - colFrom) + 1, dest);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD selectSubmatrix(int rowFrom, int colFrom, int rowTo, int colTo) {
        checkSubmatrixIndexes(rowFrom, colFrom, rowTo, colTo);
        MatrixD copy = create(rowTo - rowFrom + 1, colTo - colFrom + 1);
        return submatrix(rowFrom, colFrom, rowTo, colTo, copy, 0, 0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD appendColumn(MatrixD colVector) {
        Checks.checkCommensurateColVector(this, colVector);
        double[] _a = a;
        double[] _b = colVector.getArrayUnsafe();
        double[] _ab = new double[rows * (cols + 1)];
        System.arraycopy(_a, 0, _ab, 0, _a.length);
        System.arraycopy(_b, 0, _ab, _a.length, _b.length);
        return create(rows, cols + 1, _ab);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD appendMatrix(MatrixD matrix) {
        Checks.checkSameRows(this, matrix);
        int colsNew = cols + matrix.numColumns();
        double[] _a = a;
        double[] _b = matrix.getArrayUnsafe();
        double[] _ab = new double[rows * colsNew];
        System.arraycopy(_a, 0, _ab, 0, _a.length);
        System.arraycopy(_b, 0, _ab, _a.length, _b.length);
        return create(rows, colsNew, _ab);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD mldivide(MatrixD B) {
        Checks.checkSameRows(this, B);
        return solve(B, Matrices.createD(cols, B.numColumns()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD mrdivide(MatrixD B) {
        Checks.checkSameCols(this, B);
        return B.transpose().mldivide(this.transpose()).transpose();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD times(MatrixD B) {
        return mult(B, create(rows, B.numColumns()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD timesTimes(MatrixD B, MatrixD C) {
        return mult(B, create(rows, B.numColumns())).mult(C, create(rows, C.numColumns()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD times(ComplexMatrixD B) {
        Checks.checkMult(this, B);
        ComplexMatrixD Ac = Matrices.convertToComplex(this);
        return Ac.times(B);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD transpose() {
        return trans(create(cols, rows));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD inverse() {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("The inverse is only defined for square matrices");
        }
        return inv(create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD plus(MatrixD B) {
        return add(B, create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD timesPlus(MatrixD B, MatrixD C) {
        return multAdd(B, C.copy());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD minus(MatrixD B) {
        return add(-1.0, B, create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD uminus() {
        return scale(-1.0, create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD abs() {
        MatrixD m = copy();
        double[] b_ = m.getArrayUnsafe();
        for (int i = 0; i < b_.length; ++i) {
            double x = b_[i]; 
            if (x < 0.0) {
                b_[i] = -x;
            }
        }
        return m;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD reshape(int rows, int cols) {
        Checks.checkCompatibleDimension(this, rows, cols);
        return create(rows, cols, Arrays.copyOf(a, a.length));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[] singularValues() {
        return svd(false).getS();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD toComplexMatrix() {
        return Matrices.convertToComplex(this);
    }

    // protected methods

    protected abstract MatrixD create(int rows, int cols);

    protected abstract MatrixD create(int rows, int cols, double[] data);

    protected static void checkArrayLength(double[] array, int rows, int cols) {
        if (array.length != rows * cols) {
            throw new IllegalArgumentException(
                    "data array has wrong length. Needed : " + rows * cols + " , Is : " + array.length);
        }
    }
}
