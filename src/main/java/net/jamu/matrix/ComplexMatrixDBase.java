/*
 * Copyright 2020 Stefan Zobel
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
            double aip1 = _a[i + 1];
            _a[i] = ai * alphar - aip1 * alphai;
            _a[i + 1] = ai * alphai + aip1 * alphar;
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
            _b[i + 1] = ai * alphai + aip1 * alphar;
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
                double bip1 = _b[i + 1];
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
                _c[i + 1] = _a[i + 1] + (bi * alphai + bip1 * alphar);
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
        return setInplace(1.0, 0.0, other);
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
            double bip1 = _b[i + 1];
            _a[i] = bi * alphar - bip1 * alphai;
            _a[i + 1] = bi * alphai + bip1 * alphar;
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

    // TODO ...

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

    // TODO ...

    // DComplexMatrixBasicOps

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
        return mult(B, create(rows, B.numColumns())).mult(C, create(rows, C.numColumns()));
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
    public ComplexMatrixD conjugateTranspose() {
        return conjTrans(create(cols, rows));
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
