/*
 * Copyright 2019, 2020 Stefan Zobel
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

import net.dedekind.blas.Blas;
import net.frobenius.TTrans;
import net.frobenius.lapack.PlainLapack;

/**
 * A simple dense matrix implementation of a column-major layout double matrix
 * based on {@code BLAS} and {@code LAPACK} routines.
 */
public class SimpleMatrixF extends MatrixFBase implements MatrixF {

    private static final float BETA = 1.0f;

    /**
     * Create a new {@code SimpleMatrixF} of dimension {@code (rows, cols)}.
     * 
     * @param rows
     *            number of matrix rows
     * @param cols
     *            number of matrix columns
     */
    public SimpleMatrixF(int rows, int cols) {
        this(rows, cols, new float[Checks.checkArrayLength(rows, cols)]);
    }

    /**
     * Create a new {@code SimpleMatrixF} of dimension {@code (rows, cols)} with
     * all matrix elements set to {@code initialValue}.
     * 
     * @param rows
     *            number of matrix rows
     * @param cols
     *            number of matrix columns
     * @param initialValue
     *            the initial value to set
     */
    public SimpleMatrixF(int rows, int cols, float initialValue) {
        super(rows, cols, new float[Checks.checkArrayLength(rows, cols)], false);
        Arrays.fill(a, initialValue);
    }

    private SimpleMatrixF(SimpleMatrixF other) {
        super(other.rows, other.cols, other.a, true);
    }

    protected SimpleMatrixF(int rows, int cols, float[] data) {
        super(rows, cols, data, false);
    }

    @Override
    protected MatrixF create(int rows, int cols) {
        return new SimpleMatrixF(rows, cols);
    }

    @Override
    protected MatrixF create(int rows, int cols, float[] data) {
        return new SimpleMatrixF(rows, cols, data);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF multAdd(float alpha, MatrixF B, MatrixF C) {
        Checks.checkMultAdd(this, B, C);

        Blas blas = Matrices.getBlas();
        blas.sgemm(TTrans.NO_TRANS.val(), TTrans.NO_TRANS.val(), C.numRows(), C.numColumns(), cols, alpha, a,
                Math.max(1, rows), B.getArrayUnsafe(), Math.max(1, B.numRows()), BETA, C.getArrayUnsafe(),
                Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transABmultAdd(float alpha, MatrixF B, MatrixF C) {
        Checks.checkTransABmultAdd(this, B, C);

        Blas blas = Matrices.getBlas();
        blas.sgemm(TTrans.TRANS.val(), TTrans.TRANS.val(), C.numRows(), C.numColumns(), rows, alpha, a,
                Math.max(1, rows), B.getArrayUnsafe(), Math.max(1, B.numRows()), BETA, C.getArrayUnsafe(),
                Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transAmultAdd(float alpha, MatrixF B, MatrixF C) {
        Checks.checkTransAmultAdd(this, B, C);

        Blas blas = Matrices.getBlas();
        blas.sgemm(TTrans.TRANS.val(), TTrans.NO_TRANS.val(), C.numRows(), C.numColumns(), rows, alpha, a,
                Math.max(1, rows), B.getArrayUnsafe(), Math.max(1, B.numRows()), BETA, C.getArrayUnsafe(),
                Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transBmultAdd(float alpha, MatrixF B, MatrixF C) {
        Checks.checkTransBmultAdd(this, B, C);

        Blas blas = Matrices.getBlas();
        blas.sgemm(TTrans.NO_TRANS.val(), TTrans.TRANS.val(), C.numRows(), C.numColumns(), cols, alpha, a,
                Math.max(1, rows), B.getArrayUnsafe(), Math.max(1, B.numRows()), BETA, C.getArrayUnsafe(),
                Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF solve(MatrixF B, MatrixF X) {
        Checks.checkSolve(this, B, X);
        if (this.isSquareMatrix()) {
            return lusolve(this, X, B);
        }
        return qrsolve(this, X, B);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public SvdF svd(boolean full) {
        return new SvdF(this, full);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public SvdEconF svdEcon() {
        return new SvdEconF(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public EvdF evd(boolean full) {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("EVD only works for square matrices");
        }
        return new EvdF(this, full);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public QrdF qrd() {
        return new QrdF(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public LudF lud() {
        return new LudF(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float norm2() {
        return new SvdF(this, false).norm2();
    }

    private static MatrixF lusolve(MatrixF A, MatrixF X, MatrixF B) {
        X.setInplace(B);
        PlainLapack.sgesv(Matrices.getLapack(), A.numRows(), B.numColumns(), A.getArrayUnsafe().clone(),
                Math.max(1, A.numRows()), new int[A.numRows()], X.getArrayUnsafe(), Math.max(1, A.numRows()));
        return X;
    }

    private static MatrixF qrsolve(MatrixF A, MatrixF X, MatrixF B) {
        int rhsCount = B.numColumns();
        int mm = A.numRows();
        int nn = A.numColumns();

        SimpleMatrixF tmp = new SimpleMatrixF(Math.max(mm, nn), rhsCount);
        for (int j = 0; j < rhsCount; ++j) {
            for (int i = 0; i < mm; ++i) {
                tmp.setUnsafe(i, j, B.getUnsafe(i, j));
            }
        }

        PlainLapack.sgels(Matrices.getLapack(), TTrans.NO_TRANS, mm, nn, rhsCount, A.getArrayUnsafe().clone(),
                Math.max(1, mm), tmp.getArrayUnsafe(), Math.max(1, Math.max(mm, nn)));

        for (int j = 0; j < rhsCount; ++j) {
            for (int i = 0; i < nn; ++i) {
                X.setUnsafe(i, j, tmp.getUnsafe(i, j));
            }
        }
        return X;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF copy() {
        return new SimpleMatrixF(this);
    }
}
