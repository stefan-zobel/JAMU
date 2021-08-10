/*
 * Copyright 2020, 2021 Stefan Zobel
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
import java.util.Objects;

import net.jamu.complex.ZArrayUtil;
import net.jamu.complex.Zd;
import net.jamu.complex.ZdImpl;

/**
 * A {@code ComplexMatrixDBase} is a partial implementation of a dense matrix of
 * double precision complex numbers represented as an array of primitive doubles
 * with column-major storage layout. The addressing is zero based. All
 * operations throw a {@code NullPointerException} if any of the method
 * arguments is {@code null}.
 */
public abstract class ComplexMatrixDBase extends DimensionsBase implements ComplexMatrixD {

    protected final double[] a;

    public ComplexMatrixDBase(int rows, int cols, double[] array, boolean doArrayCopy) {
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
    public Zd toScalar() {
        if (!isScalar()) {
            throw new IllegalStateException("(" + rows + " x " + cols + ") matrix is not a scalar");
        }
        return new ZdImpl(a[0], a[1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD scaleInplace(double alphar, double alphai) {
        if (alphar == 0.0 && alphai == 0.0) {
            return this.zeroInplace();
        }
        if (alphar == 1.0 && alphai == 0.0) {
            return this;
        }
        double[] _a = a;
        for (int i = 0; i < _a.length; i += 2) {
            double ai = _a[i];
            double aip1 = _a[i + 1]; // "lgtm[java/index-out-of-bounds]"
            _a[i] = ai * alphar - aip1 * alphai;
            _a[i + 1] = ai * alphai + aip1 * alphar; // "lgtm[java/index-out-of-bounds]"
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD scale(double alphar, double alphai, ComplexMatrixD B) {
        Checks.checkEqualDimension(this, B);
        if (alphar == 0.0 && alphai == 0.0) {
            Arrays.fill(B.getArrayUnsafe(), 0.0);
            return B;
        }
        double[] _a = a;
        double[] _b = B.getArrayUnsafe();
        for (int i = 0; i < _b.length; i += 2) {
            double ai = _a[i];
            double aip1 = _a[i + 1];
            _b[i] = ai * alphar - aip1 * alphai;
            _b[i + 1] = ai * alphai + aip1 * alphar; // "lgtm[java/index-out-of-bounds]"
        }
        return B;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjTrans(ComplexMatrixD AH) {
        Checks.checkTrans(this, AH);
        int cols_ = cols;
        int rows_ = rows;
        double[] _a = a;
        double[] _ah = AH.getArrayUnsafe();
        DimensionsBase B = (DimensionsBase) AH;
        for (int col = 0; col < cols_; ++col) {
            for (int row = 0; row < rows_; ++row) {
                int aidx = 2 * idx(row, col);
                int bidx = 2 * B.idx(col, row);
                _ah[bidx] = _a[aidx];
                _ah[bidx + 1] = -_a[aidx + 1];
            }
        }
        return AH;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD trans(ComplexMatrixD AT) {
        Checks.checkTrans(this, AT);
        int cols_ = cols;
        int rows_ = rows;
        double[] _a = a;
        double[] _at = AT.getArrayUnsafe();
        DimensionsBase B = (DimensionsBase) AT;
        for (int col = 0; col < cols_; ++col) {
            for (int row = 0; row < rows_; ++row) {
                int aidx = 2 * idx(row, col);
                int bidx = 2 * B.idx(col, row);
                _at[bidx] = _a[aidx];
                _at[bidx + 1] = _a[aidx + 1];
            }
        }
        return AT;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD addInplace(ComplexMatrixD B) {
        return addInplace(1.0, 0.0, B);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD addInplace(double alphar, double alphai, ComplexMatrixD B) {
        Checks.checkEqualDimension(this, B);
        if (alphar != 0.0 && alphai != 0.0) {
            double[] _a = a;
            double[] _b = B.getArrayUnsafe();
            for (int i = 0; i < _b.length; i += 2) {
                double bi = _b[i];
                double bip1 = _b[i + 1]; // "lgtm[java/index-out-of-bounds]"
                _a[i] = bi * alphar - bip1 * alphai;
                _a[i + 1] = bi * alphai + bip1 * alphar;
            }
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD add(ComplexMatrixD B, ComplexMatrixD C) {
        return add(1.0, 0.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD add(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        Checks.checkAdd(this, B, C);
        if (alphar == 0.0 && alphai == 0.0) {
            System.arraycopy(a, 0, C.getArrayUnsafe(), 0, a.length);
        } else {
            double[] _a = a;
            double[] _b = B.getArrayUnsafe();
            double[] _c = C.getArrayUnsafe();
            for (int i = 0; i < _a.length; i += 2) {
                double bi = _b[i];
                double bip1 = _b[i + 1];
                _c[i] = _a[i] + (bi * alphar - bip1 * alphai);
                _c[i + 1] = _a[i + 1] + (bi * alphai + bip1 * alphar); // "lgtm[java/index-out-of-bounds]"
            }
        }
        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD mult(ComplexMatrixD B, ComplexMatrixD C) {
        return mult(1.0, 0.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD mult(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        return multAdd(alphar, alphai, B, C.zeroInplace());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD multAdd(ComplexMatrixD B, ComplexMatrixD C) {
        return multAdd(1.0, 0.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract ComplexMatrixD multAdd(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C);

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjTransABmult(ComplexMatrixD B, ComplexMatrixD C) {
        return conjTransABmult(1.0, 0.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjTransABmult(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        return conjTransABmultAdd(alphar, alphai, B, C.zeroInplace());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjTransAmult(ComplexMatrixD B, ComplexMatrixD C) {
        return conjTransAmult(1.0, 0.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjTransAmult(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        return conjTransAmultAdd(alphar, alphai, B, C.zeroInplace());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjTransBmult(ComplexMatrixD B, ComplexMatrixD C) {
        return conjTransBmult(1.0, 0.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjTransBmult(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        return conjTransBmultAdd(alphar, alphai, B, C.zeroInplace());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjTransABmultAdd(ComplexMatrixD B, ComplexMatrixD C) {
        return conjTransABmultAdd(1.0, 0.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract ComplexMatrixD conjTransABmultAdd(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C);

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjTransAmultAdd(ComplexMatrixD B, ComplexMatrixD C) {
        return conjTransAmultAdd(1.0, 0.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract ComplexMatrixD conjTransAmultAdd(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C);

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjTransBmultAdd(ComplexMatrixD B, ComplexMatrixD C) {
        return conjTransBmultAdd(1.0, 0.0, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract ComplexMatrixD conjTransBmultAdd(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C);

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD zeroInplace() {
        Arrays.fill(a, 0.0);
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD setInplace(ComplexMatrixD other) {
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
    public ComplexMatrixD setInplace(double alphar, double alphai, ComplexMatrixD other) {
        Checks.checkEqualDimension(this, other);
        if (alphar == 0.0 && alphai == 0.0) {
            return zeroInplace();
        }
        if (other == this) {
            return scaleInplace(alphar, alphai);
        }
        double[] _a = a;
        double[] _b = other.getArrayUnsafe();
        for (int i = 0; i < _b.length; i += 2) {
            double bi = _b[i];
            double bip1 = _b[i + 1]; // "lgtm[java/index-out-of-bounds]"
            _a[i] = bi * alphar - bip1 * alphai;
            _a[i + 1] = bi * alphai + bip1 * alphar;
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD setInplaceUpperTrapezoidal(ComplexMatrixD B) {
        Checks.checkB_hasAtLeastAsManyColsAsA(this, B);
        Checks.checkB_hasAtLeastAsManyRowsAsARowColMin(this, B);
        int cols_ = cols;
        int rows_ = rows;
        ZdImpl entry = new ZdImpl(0.0);
        for (int col = 0; col < cols_; ++col) {
            for (int row = 0; row < rows_; ++row) {
                if (row <= col) {
                    B.getUnsafe(row, col, entry);
                    this.setUnsafe(row, col, entry.re(), entry.im());
                } else {
                    this.setUnsafe(row, col, 0.0, 0.0);
                }
            }
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD setInplaceLowerTrapezoidal(ComplexMatrixD B) {
        Checks.checkB_hasAtLeastAsManyRowsAsA(this, B);
        Checks.checkB_hasAtLeastAsManyColsAsARowColMin(this, B);
        int cols_ = cols;
        int rows_ = rows;
        ZdImpl entry = new ZdImpl(0.0);
        for (int col = 0; col < cols_; ++col) {
            for (int row = 0; row < rows_; ++row) {
                if (row >= col) {
                    B.getUnsafe(row, col, entry);
                    this.setUnsafe(row, col, entry.re(), entry.im());
                } else {
                    this.setUnsafe(row, col, 0.0, 0.0);
                }
            }
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD submatrix(int r0, int c0, int r1, int c1, ComplexMatrixD B, int rb, int cb) {
        checkSubmatrixIndexes(r0, c0, r1, c1);
        B.checkIndex(rb, cb);
        B.checkIndex(rb + r1 - r0, cb + c1 - c0);
        double[] _a = a;
        double[] _b = B.getArrayUnsafe();
        int rbStart = rb;
        DimensionsBase BB = (DimensionsBase) B;
        for (int col = c0; col <= c1; ++col) {
            for (int row = r0; row <= r1; ++row) {
                int bidx = 2 * BB.idx(rb++, cb);
                int aidx = 2 * idx(row, col);
                _b[bidx] = _a[aidx];
                _b[bidx + 1] = _a[aidx + 1];
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
    public ComplexMatrixD setSubmatrixInplace(int r0, int c0, ComplexMatrixD B, int rb0, int cb0, int rb1, int cb1) {
        B.checkSubmatrixIndexes(rb0, cb0, rb1, cb1);
        checkIndex(r0, c0);
        checkIndex(r0 + rb1 - rb0, c0 + cb1 - cb0);
        double[] _a = a;
        double[] _b = B.getArrayUnsafe();
        int r0Start = r0;
        DimensionsBase BB = (DimensionsBase) B;
        for (int col = cb0; col <= cb1; ++col) {
            for (int row = rb0; row <= rb1; ++row) {
                int bidx = 2 * BB.idx(row, col);
                int aidx = 2 * idx(r0++, c0);
                _a[aidx] = _b[bidx];
                _a[aidx + 1] = _b[bidx + 1];
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
    public void get(int row, int col, Zd out) {
        Objects.requireNonNull(out);
        checkIndex(row, col);
        int idx = 2 * idx(row, col);
        out.set(a[idx], a[idx + 1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Zd get(int row, int col) {
        checkIndex(row, col);
        int idx = 2 * idx(row, col);
        return new ZdImpl(a[idx], a[idx + 1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD set(int row, int col, double valr, double vali) {
        checkIndex(row, col);
        int idx = 2 * idx(row, col);
        a[idx] = valr;
        a[idx + 1] = vali;
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD add(int row, int col, double valr, double vali) {
        checkIndex(row, col);
        int idx = 2 * idx(row, col);
        a[idx] += valr;
        a[idx + 1] += vali;
        return this;
    }

    protected void addUnsafe(int row, int col, double valr, double vali) {
        int idx = 2 * idx(row, col);
        a[idx] += valr;
        a[idx + 1] += vali;
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
    public void getUnsafe(int row, int col, Zd out) {
        int idx = 2 * idx(row, col);
        out.set(a[idx], a[idx + 1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Zd getUnsafe(int row, int col) {
        int idx = 2 * idx(row, col);
        return new ZdImpl(a[idx], a[idx + 1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setUnsafe(int row, int col, double valr, double vali) {
        int idx = 2 * idx(row, col);
        a[idx] = valr;
        a[idx + 1] = vali;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD inv(ComplexMatrixD inverse) {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("The inverse is only defined for square matrices");
        }
        Checks.checkEqualDimension(this, inverse);
        return solve(Matrices.identityComplexD(this.numRows()), inverse);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD pseudoInv() {
        if (this.isSquareMatrix()) {
            return inv(create(rows, cols));
        }
        SvdComplexD svd = svd(true);
        double tol = MACH_EPS_DBL * Math.max(rows, cols) * svd.norm2();
        double[] sigma = svd.getS();
        // compute Sigma dagger (= SInv)
        ComplexMatrixD SInv = create(cols, rows);
        for (int i = 0; i < sigma.length; ++i) {
            if (sigma[i] > tol) {
                SInv.setUnsafe(i, i, 1.0 / sigma[i], 0.0);
            }
        }
        // Vh conjugate-transposed (= Vh*) times Sigma dagger
        ComplexMatrixD Vh = svd.getVh();
        ComplexMatrixD x = Vh.conjTransAmult(SInv, create(Vh.numRows(), SInv.numColumns()));
        // compute x times U conjugate-transposed (= xU*)
        ComplexMatrixD U = svd.getU();
        // voila, the Moore-Penrose pseudoinverse
        return x.conjTransBmult(U, create(x.numRows(), U.numRows()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD expm() {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("Matrix exponentiation is only defined for square matrices");
        }
        return Expm.expmComplexD(this, normMaxAbs());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[][] toJaggedArray() {
        int _rows = rows;
        int _cols = cols;
        double[] _a = a;
        double[][] copy = new double[_rows][2 * _cols];
        for (int row = 0; row < _rows; ++row) {
            double[] row_i = copy[row];
            for (int col = 0; col < _cols; ++col) {
                int idx = 2 * idx(row, col);
                int rowIdx = 2 * col;
                row_i[rowIdx] = _a[idx];
                row_i[rowIdx + 1] = _a[idx + 1];
            }
        }
        return copy;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double normF() {
        return ZArrayUtil.l2norm(a);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double normMaxAbs() {
        double max = Double.NEGATIVE_INFINITY;
        double[] _a = a;
        for (int i = 0; i < _a.length; i += 2) {
            double re = _a[i];
            double im = _a[i + 1]; // "lgtm[java/index-out-of-bounds]"
            double abs = (im == 0.0) ? Math.abs(re) : ZdImpl.abs(re, im);
            if (abs > max) {
                max = abs;
            }
        }
        return max;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double normInf() {
        ZdImpl z = new ZdImpl(0.0);
        double max = 0.0f;
        int rows_ = rows;
        int cols_ = cols;
        for (int i = 0; i < rows_; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols_; j++) {
                getUnsafe(i, j, z);
                double re = z.re();
                double im = z.im();
                double abs = (im == 0.0) ? Math.abs(re) : ZdImpl.abs(re, im);
                sum += abs;
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
            for (int idx = 2 * col * rows_; idx < 2 * (col + 1) * rows_; idx += 2) {
                double re = _a[idx];
                double im = _a[idx + 1];
                double abs = (im == 0.0) ? Math.abs(re) : ZdImpl.abs(re, im);
                sum += abs;
            }
            max = Math.max(max, sum);
        }
        return max;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Zd trace() {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("The trace of a matrix is only defined for square matrices");
        }
        Zd t = new ZdImpl(0.0);
        double re = 0.0;
        double im = 0.0;
        for (int i = 0; i < rows; ++i) {
            getUnsafe(i, i, t);
            re += t.re();
            im += t.im();
        }
        t.set(re, im);
        return t;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD zeroizeSubEpsilonInplace(int k) {
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
    public ComplexMatrixD sanitizeNonFiniteInplace(double nanSurrogate, double posInfSurrogate,
            double negInfSurrogate) {
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
    public ComplexMatrixD sanitizeNaNInplace(double nanSurrogate) {
        return sanitizeNonFiniteInplace(nanSurrogate, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD centerInplace() {
        double[] _a = a;
        int rows_ = rows;
        int cols_ = cols;
        for (int col = 0; col < cols_; ++col) {
            double reColSum = 0.0;
            double imColSum = 0.0;
            double reColMean = 0.0;
            double imColMean = 0.0;
            for (int idx = 2 * col * rows_; idx < 2 * (col + 1) * rows_; idx += 2) {
                reColSum += _a[idx];
                imColSum += _a[idx + 1];
            }
            reColMean = reColSum / rows_;
            imColMean = imColSum / rows_;
            for (int idx = 2 * col * rows_; idx < 2 * (col + 1) * rows_; idx += 2) {
                _a[idx] -= reColMean;
                _a[idx + 1] -= imColMean;
            }
        }
        return this;
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

    // DComplexMatrixBasicOps

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD selectColumn(int col) {
        return selectConsecutiveColumns(col, col);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD selectConsecutiveColumns(int colFrom, int colTo) {
        checkSubmatrixIndexes(0, colFrom, rows - 1, colTo);
        int startPos = 2 * rows * colFrom;
        int length = 2 * ((colTo - colFrom) + 1) * rows;
        double[] dest = new double[length];
        System.arraycopy(a, startPos, dest, 0, length);
        return create(rows, (colTo - colFrom) + 1, dest);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD selectSubmatrix(int rowFrom, int colFrom, int rowTo, int colTo) {
        checkSubmatrixIndexes(rowFrom, colFrom, rowTo, colTo);
        ComplexMatrixD copy = create(rowTo - rowFrom + 1, colTo - colFrom + 1);
        return submatrix(rowFrom, colFrom, rowTo, colTo, copy, 0, 0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD appendColumn(ComplexMatrixD colVector) {
        Checks.checkCommensurateColVector(this, colVector);
        double[] _a = a;
        double[] _b = colVector.getArrayUnsafe();
        double[] _ab = new double[2 * (rows * (cols + 1))];
        System.arraycopy(_a, 0, _ab, 0, _a.length);
        System.arraycopy(_b, 0, _ab, _a.length, _b.length);
        return create(rows, cols + 1, _ab);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD appendMatrix(ComplexMatrixD matrix) {
        Checks.checkSameRows(this, matrix);
        int colsNew = cols + matrix.numColumns();
        double[] _a = a;
        double[] _b = matrix.getArrayUnsafe();
        double[] _ab = new double[2 * (rows * colsNew)];
        System.arraycopy(_a, 0, _ab, 0, _a.length);
        System.arraycopy(_b, 0, _ab, _a.length, _b.length);
        return create(rows, colsNew, _ab);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD mldivide(ComplexMatrixD B) {
        Checks.checkSameRows(this, B);
        return solve(B, Matrices.createComplexD(cols, B.numColumns()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD mrdivide(ComplexMatrixD B) {
        Checks.checkSameCols(this, B);
        return B.conjugateTranspose().mldivide(this.conjugateTranspose()).conjugateTranspose();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD times(ComplexMatrixD B) {
        return mult(B, create(rows, B.numColumns()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD timesTimes(ComplexMatrixD B, ComplexMatrixD C) {
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
    public ComplexMatrixD timesMany(ComplexMatrixD m, ComplexMatrixD... matrices) {
        int len = matrices.length;
        if (len == 0) {
            return times(m);
        }
        if (len == 1) {
            return timesTimes(m, matrices[0]);
        }
        Checks.checkMultMany(this, m, matrices);
        UPInts splits = new MatrixChain(this, m, matrices).computeSplits();
        ArrayList<ComplexMatrixD> chain = MatrixChain.buildList(this, m, matrices);
        return multiplyMany(0, chain.size() - 1, chain, splits);
    }

    /*
     * Recursive function for the actual matrix-chain multiplication, also from
     * Cormen et al.
     */
    private ComplexMatrixD multiplyMany(int i, int j, ArrayList<ComplexMatrixD> chain, UPInts splits) {
        if (i == j) {
            // base case
            return chain.get(i);
        }
        int k = splits.get(i, j);
        ComplexMatrixD X = multiplyMany(i, k, chain, splits);
        ComplexMatrixD Y = multiplyMany(k + 1, j, chain, splits);
        return X.mult(Y, create(X.numRows(), Y.numColumns()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD timesConjugateTransposed() {
        return conjTransBmult(this, create(rows, rows));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjugateTransposedTimes() {
        return conjTransAmult(this, create(cols, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD times(MatrixD B) {
        Checks.checkMult(this, B);
        ComplexMatrixD Bc = Matrices.convertToComplex(B);
        return this.times(Bc);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD plus(ComplexMatrixD B) {
        return add(B, create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD timesPlus(ComplexMatrixD B, ComplexMatrixD C) {
        return multAdd(B, C.copy());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD timesMinus(ComplexMatrixD B, ComplexMatrixD C) {
        return multAdd(B, C.uminus());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD minus(ComplexMatrixD B) {
        return add(-1.0, 0.0, B, create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD uminus() {
        return scale(-1.0, 0.0, create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD abs() {
        ComplexMatrixD m = copy();
        double[] b_ = m.getArrayUnsafe();
        for (int i = 0; i < b_.length; i += 2) {
            double re = b_[i];
            double im = b_[i + 1]; // "lgtm[java/index-out-of-bounds]"
            // nano-optimize
            if (im == 0.0) {
                if (re < 0.0) {
                    b_[i] = -re;
                }
            } else {
                b_[i] = ZdImpl.abs(re, im);
                b_[i + 1] = 0.0; // "lgtm[java/index-out-of-bounds]"
            }
        }
        return m;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjugateTranspose() {
        return conjTrans(create(cols, rows));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD transpose() {
        return trans(create(cols, rows));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD inverse() {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("The inverse is only defined for square matrices");
        }
        return inv(create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD reshape(int rows, int cols) {
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
    public MatrixD toRealMatrix() {
        return Matrices.convertToReal(this);
    }

    // protected methods

    protected abstract ComplexMatrixD create(int rows, int cols);

    protected abstract ComplexMatrixD create(int rows, int cols, double[] data);

    protected static void checkArrayLength(double[] array, int rows, int cols) {
        if (array.length != 2 * (rows * cols)) {
            throw new IllegalArgumentException(
                    "data array has wrong length. Needed : " + 2 * (rows * cols) + " , Is : " + array.length);
        }
    }
}
