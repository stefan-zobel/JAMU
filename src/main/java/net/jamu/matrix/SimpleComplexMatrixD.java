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

import net.dedekind.blas.BlasExt;
import net.dedekind.blas.Trans;
import net.dedekind.lapack.Lapack;
import net.frobenius.TTrans;
import net.frobenius.lapack.PlainLapack;
import net.jamu.complex.Zd;
import net.jamu.complex.ZdImpl;

/**
 * A simple dense matrix implementation of a column-major layout double
 * precision complex matrix based on {@code BLAS} and {@code LAPACK} routines.
 */
public class SimpleComplexMatrixD extends ComplexMatrixDBase implements ComplexMatrixD {

    private static final double BETA_R = 1.0;
    private static final double BETA_I = 0.0;

    /**
     * Create a new {@code SimpleComplexMatrixD} of dimension
     * {@code (rows, cols)}.
     * 
     * @param rows
     *            number of matrix rows
     * @param cols
     *            number of matrix columns
     */
    public SimpleComplexMatrixD(int rows, int cols) {
        this(rows, cols, new double[Checks.checkComplexArrayLength(rows, cols)]);
    }

    /**
     * Create a new {@code SimpleComplexMatrixD} of dimension
     * {@code (rows, cols)} with all matrix elements set to
     * {@code initialValue}.
     * 
     * @param rows
     *            number of matrix rows
     * @param cols
     *            number of matrix columns
     * @param initialValue
     *            the initial value to set
     */
    public SimpleComplexMatrixD(int rows, int cols, double initialValue) {
        super(rows, cols, new double[Checks.checkComplexArrayLength(rows, cols)], false);
        Arrays.fill(a, initialValue);
    }

    /**
     * Create a new {@code SimpleComplexMatrixD} of dimension
     * {@code (rows, cols)} with all matrix elements set to
     * {@code (iniValr, iniVali)}.
     * 
     * @param rows
     *            number of matrix rows
     * @param cols
     *            number of matrix columns
     * @param iniValr
     *            the real part of the initial value to set
     * @param iniVali
     *            the imaginary part of the initial value to set
     */
    public SimpleComplexMatrixD(int rows, int cols, double iniValr, double iniVali) {
        super(rows, cols, new double[Checks.checkComplexArrayLength(rows, cols)], false);
        double[] a_ = a;
        for (int i = 0; i < a_.length; i += 2) {
            a_[i] = iniValr;
            a_[i + 1] = iniVali;
        }
    }

    private SimpleComplexMatrixD(SimpleComplexMatrixD other) {
        super(other.rows, other.cols, other.a, true);
    }

    protected SimpleComplexMatrixD(int rows, int cols, double[] data) {
        super(rows, cols, data, false);
    }

    @Override
    protected ComplexMatrixD create(int rows, int cols) {
        return new SimpleComplexMatrixD(rows, cols);
    }

    @Override
    protected ComplexMatrixD create(int rows, int cols, double[] data) {
        return new SimpleComplexMatrixD(rows, cols, data);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD multAdd(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        Checks.checkMultAdd(this, B, C);

        BlasExt blas = BlasExt.getInstance();
        blas.zgemm3m(Trans.N, Trans.N, C.numRows(), C.numColumns(), cols, alphar, alphai, a, Math.max(1, rows),
                B.getArrayUnsafe(), Math.max(1, B.numRows()), BETA_R, BETA_I, C.getArrayUnsafe(),
                Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjTransABmultAdd(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        Checks.checkTransABmultAdd(this, B, C);

        BlasExt blas = BlasExt.getInstance();
        blas.zgemm3m(Trans.C, Trans.C, C.numRows(), C.numColumns(), rows, alphar, alphai, a, Math.max(1, rows),
                B.getArrayUnsafe(), Math.max(1, B.numRows()), BETA_R, BETA_I, C.getArrayUnsafe(),
                Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjTransAmultAdd(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        Checks.checkTransAmultAdd(this, B, C);

        BlasExt blas = BlasExt.getInstance();
        blas.zgemm3m(Trans.C, Trans.N, C.numRows(), C.numColumns(), rows, alphar, alphai, a, Math.max(1, rows),
                B.getArrayUnsafe(), Math.max(1, B.numRows()), BETA_R, BETA_I, C.getArrayUnsafe(),
                Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjTransBmultAdd(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        Checks.checkTransBmultAdd(this, B, C);

        BlasExt blas = BlasExt.getInstance();
        blas.zgemm3m(Trans.N, Trans.C, C.numRows(), C.numColumns(), cols, alphar, alphai, a, Math.max(1, rows),
                B.getArrayUnsafe(), Math.max(1, B.numRows()), BETA_R, BETA_I, C.getArrayUnsafe(),
                Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD solve(ComplexMatrixD B, ComplexMatrixD X) {
        Checks.checkSolve(this, B, X);
        if (this.isSquareMatrix()) {
            return lusolve(this, X, B);
        }
        return qrsolve(this, X, B);
    }

    // TODO ...

    /**
     * {@inheritDoc}
     */
    @Override
    public EvdComplexD evd(boolean full) {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("EVD only works for square matrices");
        }
        return new EvdComplexD(this, full);
    }

    // TODO ...

    private static ComplexMatrixD lusolve(ComplexMatrixD A, ComplexMatrixD X, ComplexMatrixD B) {
        X.setInplace(B);
        PlainLapack.zgesv(Lapack.getInstance(), A.numRows(), B.numColumns(), A.getArrayUnsafe().clone(),
                Math.max(1, A.numRows()), new int[A.numRows()], X.getArrayUnsafe(), Math.max(1, A.numRows()));
        return X;
    }

    private static ComplexMatrixD qrsolve(ComplexMatrixD A, ComplexMatrixD X, ComplexMatrixD B) {
        int rhsCount = B.numColumns();
        int mm = A.numRows();
        int nn = A.numColumns();

        SimpleComplexMatrixD tmp = new SimpleComplexMatrixD(Math.max(mm, nn), rhsCount);
        Zd zVal = new ZdImpl(0.0);
        for (int j = 0; j < rhsCount; ++j) {
            for (int i = 0; i < mm; ++i) {
                B.getUnsafe(i, j, zVal);
                tmp.setUnsafe(i, j, zVal.re(), zVal.im());
            }
        }

        PlainLapack.zgels(Lapack.getInstance(), TTrans.NO_TRANS, mm, nn, rhsCount, A.getArrayUnsafe().clone(),
                Math.max(1, mm), tmp.getArrayUnsafe(), Math.max(1, Math.max(mm, nn)));

        for (int j = 0; j < rhsCount; ++j) {
            for (int i = 0; i < nn; ++i) {
                tmp.getUnsafe(i, j, zVal);
                X.setUnsafe(i, j, zVal.re(), zVal.im());
            }
        }
        return X;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD copy() {
        return new SimpleComplexMatrixD(this);
    }
}
