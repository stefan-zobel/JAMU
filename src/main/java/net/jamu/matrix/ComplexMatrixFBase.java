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

import java.util.Arrays;
import java.util.Objects;

import net.jamu.complex.ZArrayUtil;
import net.jamu.complex.Zf;
import net.jamu.complex.ZfImpl;

/**
 * A {@code ComplexMatrixFBase} is a partial implementation of a dense matrix of
 * single precision complex numbers represented as an array of primitive floats
 * with column-major storage layout. The addressing is zero based. All
 * operations throw a {@code NullPointerException} if any of the method
 * arguments is {@code null}.
 */
public abstract class ComplexMatrixFBase extends DimensionsBase implements ComplexMatrixF {

    protected final float[] a;

    public ComplexMatrixFBase(int rows, int cols, float[] array, boolean doArrayCopy) {
        super(rows, cols);
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
    public Zf toScalar() {
        if (!isScalar()) {
            throw new IllegalStateException("(" + rows + " x " + cols + ") matrix is not a scalar");
        }
        return new ZfImpl(a[0], a[1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF scaleInplace(float alphar, float alphai) {
        if (alphar == 0.0f && alphai == 0.0f) {
            return this.zeroInplace();
        }
        if (alphar == 1.0f && alphai == 0.0f) {
            return this;
        }
        float[] _a = a;
        for (int i = 0; i < _a.length; i += 2) {
            float ai = _a[i];
            float aip1 = _a[i + 1]; // "lgtm[java/index-out-of-bounds]"
            _a[i] = ai * alphar - aip1 * alphai;
            _a[i + 1] = ai * alphai + aip1 * alphar; // "lgtm[java/index-out-of-bounds]"
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF scale(float alphar, float alphai, ComplexMatrixF B) {
        Checks.checkEqualDimension(this, B);
        if (alphar == 0.0f && alphai == 0.0f) {
            Arrays.fill(B.getArrayUnsafe(), 0.0f);
            return B;
        }
        float[] _a = a;
        float[] _b = B.getArrayUnsafe();
        for (int i = 0; i < _b.length; i += 2) {
            float ai = _a[i];
            float aip1 = _a[i + 1];
            _b[i] = ai * alphar - aip1 * alphai;
            _b[i + 1] = ai * alphai + aip1 * alphar; // "lgtm[java/index-out-of-bounds]"
        }
        return B;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF conjTrans(ComplexMatrixF AH) {
        Checks.checkTrans(this, AH);
        int cols_ = cols;
        int rows_ = rows;
        float[] _a = a;
        float[] _ah = AH.getArrayUnsafe();
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
    public ComplexMatrixF trans(ComplexMatrixF AT) {
        Checks.checkTrans(this, AT);
        int cols_ = cols;
        int rows_ = rows;
        float[] _a = a;
        float[] _at = AT.getArrayUnsafe();
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
    public ComplexMatrixF addInplace(ComplexMatrixF B) {
        return addInplace(1.0f, 0.0f, B);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF addInplace(float alphar, float alphai, ComplexMatrixF B) {
        Checks.checkEqualDimension(this, B);
        if (alphar != 0.0f && alphai != 0.0f) {
            float[] _a = a;
            float[] _b = B.getArrayUnsafe();
            for (int i = 0; i < _b.length; i += 2) {
                float bi = _b[i];
                float bip1 = _b[i + 1]; // "lgtm[java/index-out-of-bounds]"
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
    public ComplexMatrixF add(ComplexMatrixF B, ComplexMatrixF C) {
        return add(1.0f, 0.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF add(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        Checks.checkAdd(this, B, C);
        if (alphar == 0.0f && alphai == 0.0f) {
            System.arraycopy(a, 0, C.getArrayUnsafe(), 0, a.length);
        } else {
            float[] _a = a;
            float[] _b = B.getArrayUnsafe();
            float[] _c = C.getArrayUnsafe();
            for (int i = 0; i < _a.length; i += 2) {
                float bi = _b[i];
                float bip1 = _b[i + 1];
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
    public ComplexMatrixF mult(ComplexMatrixF B, ComplexMatrixF C) {
        return mult(1.0f, 0.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF mult(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        return multAdd(alphar, alphai, B, C.zeroInplace());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF multAdd(ComplexMatrixF B, ComplexMatrixF C) {
        return multAdd(1.0f, 0.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract ComplexMatrixF multAdd(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C);

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF conjTransABmult(ComplexMatrixF B, ComplexMatrixF C) {
        return conjTransABmult(1.0f, 0.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF conjTransABmult(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        return conjTransABmultAdd(alphar, alphai, B, C.zeroInplace());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF conjTransAmult(ComplexMatrixF B, ComplexMatrixF C) {
        return conjTransAmult(1.0f, 0.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF conjTransAmult(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        return conjTransAmultAdd(alphar, alphai, B, C.zeroInplace());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF conjTransBmult(ComplexMatrixF B, ComplexMatrixF C) {
        return conjTransBmult(1.0f, 0.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF conjTransBmult(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        return conjTransBmultAdd(alphar, alphai, B, C.zeroInplace());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF conjTransABmultAdd(ComplexMatrixF B, ComplexMatrixF C) {
        return conjTransABmultAdd(1.0f, 0.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract ComplexMatrixF conjTransABmultAdd(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C);

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF conjTransAmultAdd(ComplexMatrixF B, ComplexMatrixF C) {
        return conjTransAmultAdd(1.0f, 0.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract ComplexMatrixF conjTransAmultAdd(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C);

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF conjTransBmultAdd(ComplexMatrixF B, ComplexMatrixF C) {
        return conjTransBmultAdd(1.0f, 0.0f, B, C);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract ComplexMatrixF conjTransBmultAdd(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C);

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF zeroInplace() {
        Arrays.fill(a, 0.0f);
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF setInplace(ComplexMatrixF other) {
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
    public ComplexMatrixF setInplace(float alphar, float alphai, ComplexMatrixF other) {
        Checks.checkEqualDimension(this, other);
        if (alphar == 0.0f && alphai == 0.0f) {
            return zeroInplace();
        }
        if (other == this) {
            return scaleInplace(alphar, alphai);
        }
        float[] _a = a;
        float[] _b = other.getArrayUnsafe();
        for (int i = 0; i < _b.length; i += 2) {
            float bi = _b[i];
            float bip1 = _b[i + 1]; // "lgtm[java/index-out-of-bounds]"
            _a[i] = bi * alphar - bip1 * alphai;
            _a[i + 1] = bi * alphai + bip1 * alphar;
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF setInplaceUpperTrapezoidal(ComplexMatrixF B) {
        Checks.checkB_hasAtLeastAsManyColsAsA(this, B);
        Checks.checkB_hasAtLeastAsManyRowsAsARowColMin(this, B);
        int cols_ = cols;
        int rows_ = rows;
        ZfImpl entry = new ZfImpl(0.0f);
        for (int col = 0; col < cols_; ++col) {
            for (int row = 0; row < rows_; ++row) {
                if (row <= col) {
                    B.getUnsafe(row, col, entry);
                    this.setUnsafe(row, col, entry.re(), entry.im());
                } else {
                    this.setUnsafe(row, col, 0.0f, 0.0f);
                }
            }
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF setInplaceLowerTrapezoidal(ComplexMatrixF B) {
        Checks.checkB_hasAtLeastAsManyRowsAsA(this, B);
        Checks.checkB_hasAtLeastAsManyColsAsARowColMin(this, B);
        int cols_ = cols;
        int rows_ = rows;
        ZfImpl entry = new ZfImpl(0.0f);
        for (int col = 0; col < cols_; ++col) {
            for (int row = 0; row < rows_; ++row) {
                if (row >= col) {
                    B.getUnsafe(row, col, entry);
                    this.setUnsafe(row, col, entry.re(), entry.im());
                } else {
                    this.setUnsafe(row, col, 0.0f, 0.0f);
                }
            }
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF submatrix(int r0, int c0, int r1, int c1, ComplexMatrixF B, int rb, int cb) {
        checkSubmatrixIndexes(r0, c0, r1, c1);
        B.checkIndex(rb, cb);
        B.checkIndex(rb + r1 - r0, cb + c1 - c0);
        float[] _a = a;
        float[] _b = B.getArrayUnsafe();
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
    public ComplexMatrixF setSubmatrixInplace(int r0, int c0, ComplexMatrixF B, int rb0, int cb0, int rb1, int cb1) {
        B.checkSubmatrixIndexes(rb0, cb0, rb1, cb1);
        checkIndex(r0, c0);
        checkIndex(r0 + rb1 - rb0, c0 + cb1 - cb0);
        float[] _a = a;
        float[] _b = B.getArrayUnsafe();
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
    public void get(int row, int col, Zf out) {
        Objects.requireNonNull(out);
        checkIndex(row, col);
        int idx = 2 * idx(row, col);
        out.set(a[idx], a[idx + 1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Zf get(int row, int col) {
        checkIndex(row, col);
        int idx = 2 * idx(row, col);
        return new ZfImpl(a[idx], a[idx + 1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF set(int row, int col, float valr, float vali) {
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
    public ComplexMatrixF add(int row, int col, float valr, float vali) {
        checkIndex(row, col);
        int idx = 2 * idx(row, col);
        a[idx] += valr;
        a[idx + 1] += vali;
        return this;
    }

    protected void addUnsafe(int row, int col, float valr, float vali) {
        int idx = 2 * idx(row, col);
        a[idx] += valr;
        a[idx + 1] += vali;
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
    public void getUnsafe(int row, int col, Zf out) {
        int idx = 2 * idx(row, col);
        out.set(a[idx], a[idx + 1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Zf getUnsafe(int row, int col) {
        int idx = 2 * idx(row, col);
        return new ZfImpl(a[idx], a[idx + 1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setUnsafe(int row, int col, float valr, float vali) {
        int idx = 2 * idx(row, col);
        a[idx] = valr;
        a[idx + 1] = vali;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF inv(ComplexMatrixF inverse) {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("The inverse is only defined for square matrices");
        }
        Checks.checkEqualDimension(this, inverse);
        return solve(Matrices.identityComplexF(this.numRows()), inverse);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF pseudoInv() {
        if (this.isSquareMatrix()) {
            return inv(create(rows, cols));
        }
        SvdComplexF svd = svd(true);
        float tol = MACH_EPS_FLT * Math.max(rows, cols) * svd.norm2();
        float[] sigma = svd.getS();
        // compute Sigma dagger (= SInv)
        ComplexMatrixF SInv = create(cols, rows);
        for (int i = 0; i < sigma.length; ++i) {
            if (sigma[i] > tol) {
                SInv.setUnsafe(i, i, 1.0f / sigma[i], 0.0f);
            }
        }
        // x = Vh conjugate-transposed (= Vh*) times Sigma dagger
        ComplexMatrixF Vh = svd.getVh();
        ComplexMatrixF x = Vh.conjTransAmult(SInv, create(Vh.numRows(), SInv.numColumns()));
        // compute x times U conjugate-transposed (= xU*)
        ComplexMatrixF U = svd.getU();
        // voila, the Moore-Penrose pseudoinverse
        return x.conjTransBmult(U, create(x.numRows(), U.numRows()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF expm() {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("Matrix exponentiation is only defined for square matrices");
        }
        return Expm.expmComplexF(this, normMaxAbs());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float[][] toJaggedArray() {
        int _rows = rows;
        int _cols = cols;
        float[] _a = a;
        float[][] copy = new float[_rows][2 * _cols];
        for (int row = 0; row < _rows; ++row) {
            float[] row_i = copy[row];
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
    public float normF() {
        return ZArrayUtil.l2norm(a);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float normMaxAbs() {
        float max = Float.NEGATIVE_INFINITY;
        float[] _a = a;
        for (int i = 0; i < _a.length; i += 2) {
            float re = _a[i];
            float im = _a[i + 1]; // "lgtm[java/index-out-of-bounds]"
            float abs = (im == 0.0f) ? Math.abs(re) : ZfImpl.abs(re, im);
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
    public float normInf() {
        ZfImpl z = new ZfImpl(0.0f);
        double max = 0.0f;
        int rows_ = rows;
        int cols_ = cols;
        for (int i = 0; i < rows_; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols_; j++) {
                getUnsafe(i, j, z);
                float re = z.re();
                float im = z.im();
                float abs = (im == 0.0f) ? Math.abs(re) : ZfImpl.abs(re, im);
                sum += abs;
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
            for (int idx = 2 * col * rows_; idx < 2 * (col + 1) * rows_; idx += 2) {
                float re = _a[idx];
                float im = _a[idx + 1];
                float abs = (im == 0.0f) ? Math.abs(re) : ZfImpl.abs(re, im);
                sum += abs;
            }
            max = Math.max(max, sum);
        }
        return (float) max;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Zf trace() {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("The trace of a matrix is only defined for square matrices");
        }
        Zf t = new ZfImpl(0.0f);
        double re = 0.0;
        double im = 0.0;
        for (int i = 0; i < rows; ++i) {
            getUnsafe(i, i, t);
            re += t.re();
            im += t.im();
        }
        t.set((float) re, (float) im);
        return t;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF zeroizeSubEpsilonInplace(int k) {
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
    public ComplexMatrixF sanitizeNonFiniteInplace(float nanSurrogate, float posInfSurrogate, float negInfSurrogate) {
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
    public ComplexMatrixF sanitizeNaNInplace(float nanSurrogate) {
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

    // FComplexMatrixBasicOps

    /**
     * {@inheritDoc}
     */
    public ComplexMatrixF selectConsecutiveColumns(int colFrom, int colTo) {
        checkSubmatrixIndexes(0, colFrom, rows - 1, colTo);
        int startPos = 2 * rows * colFrom;
        int length = 2 * ((colTo - colFrom) + 1) * rows;
        float[] dest = new float[length];
        System.arraycopy(a, startPos, dest, 0, length);
        return create(rows, (colTo - colFrom) + 1, dest);
    }

    /**
     * {@inheritDoc}
     */
    public ComplexMatrixF selectSubmatrix(int rowFrom, int colFrom, int rowTo, int colTo) {
        checkSubmatrixIndexes(rowFrom, colFrom, rowTo, colTo);
        ComplexMatrixF copy = create(rowTo - rowFrom + 1, colTo - colFrom + 1);
        return submatrix(rowFrom, colFrom, rowTo, colTo, copy, 0, 0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF appendColumn(ComplexMatrixF colVector) {
        Checks.checkCommensurateColVector(this, colVector);
        float[] _a = a;
        float[] _b = colVector.getArrayUnsafe();
        float[] _ab = new float[2 * (rows * (cols + 1))];
        System.arraycopy(_a, 0, _ab, 0, _a.length);
        System.arraycopy(_b, 0, _ab, _a.length, _b.length);
        return create(rows, cols + 1, _ab);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF mldivide(ComplexMatrixF B) {
        Checks.checkSameRows(this, B);
        return solve(B, Matrices.createComplexF(cols, B.numColumns()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF mrdivide(ComplexMatrixF B) {
        Checks.checkSameCols(this, B);
        return B.conjugateTranspose().mldivide(this.conjugateTranspose()).conjugateTranspose();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF times(ComplexMatrixF B) {
        return mult(B, create(rows, B.numColumns()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF timesTimes(ComplexMatrixF B, ComplexMatrixF C) {
        return mult(B, create(rows, B.numColumns())).mult(C, create(rows, C.numColumns()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF times(MatrixF B) {
        Checks.checkMult(this, B);
        ComplexMatrixF Bc = Matrices.convertToComplex(B);
        return this.times(Bc);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF plus(ComplexMatrixF B) {
        return add(B, create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF timesPlus(ComplexMatrixF B, ComplexMatrixF C) {
        return multAdd(B, C.copy());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF minus(ComplexMatrixF B) {
        return add(-1.0f, 0.0f, B, create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF uminus() {
        return scale(-1.0f, 0.0f, create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF abs() {
        ComplexMatrixF m = copy();
        float[] b_ = m.getArrayUnsafe();
        for (int i = 0; i < b_.length; i += 2) {
            float re = b_[i];
            float im = b_[i + 1]; // "lgtm[java/index-out-of-bounds]"
            // nano-optimize
            if (im == 0.0f) {
                if (re < 0.0f) {
                    b_[i] = -re;
                }
            } else {
                b_[i] = ZfImpl.abs(re, im);
                b_[i + 1] = 0.0f; // "lgtm[java/index-out-of-bounds]"
            }
        }
        return m;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF conjugateTranspose() {
        return conjTrans(create(cols, rows));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF transpose() {
        return trans(create(cols, rows));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF inverse() {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("The inverse is only defined for square matrices");
        }
        return inv(create(rows, cols));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF reshape(int rows, int cols) {
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
    public MatrixF toRealMatrix() {
        return Matrices.convertToReal(this);
    }

    // protected methods

    protected abstract ComplexMatrixF create(int rows, int cols);

    protected abstract ComplexMatrixF create(int rows, int cols, float[] data);

    protected static void checkArrayLength(float[] array, int rows, int cols) {
        if (array.length != 2 * (rows * cols)) {
            throw new IllegalArgumentException(
                    "data array has wrong length. Needed : " + 2 * (rows * cols) + " , Is : " + array.length);
        }
    }
}
