/*
 * Copyright 2019, 2026 Stefan Zobel
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

        float[] c = C.getArrayUnsafe();
        ReadAccess.OperandF b = ReadAccess.operand(B, c);
        Blas blas = Matrices.getBlas();
        blas.sgemm(TTrans.NO_TRANS.val(), TTrans.NO_TRANS.val(), C.numRows(), C.numColumns(), cols, alpha, a,
                0, Math.max(1, rows), b.array, b.offset, b.ld, BETA, c, 0, Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transABmultAdd(float alpha, MatrixF B, MatrixF C) {
        Checks.checkTransABmultAdd(this, B, C);

        float[] c = C.getArrayUnsafe();
        ReadAccess.OperandF b = ReadAccess.operand(B, c);
        Blas blas = Matrices.getBlas();
        blas.sgemm(TTrans.TRANS.val(), TTrans.TRANS.val(), C.numRows(), C.numColumns(), rows, alpha, a,
                0, Math.max(1, rows), b.array, b.offset, b.ld, BETA, c, 0, Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transAmultAdd(float alpha, MatrixF B, MatrixF C) {
        Checks.checkTransAmultAdd(this, B, C);

        float[] c = C.getArrayUnsafe();
        ReadAccess.OperandF b = ReadAccess.operand(B, c);
        Blas blas = Matrices.getBlas();
        blas.sgemm(TTrans.TRANS.val(), TTrans.NO_TRANS.val(), C.numRows(), C.numColumns(), rows, alpha, a,
                0, Math.max(1, rows), b.array, b.offset, b.ld, BETA, c, 0, Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF transBmultAdd(float alpha, MatrixF B, MatrixF C) {
        Checks.checkTransBmultAdd(this, B, C);

        float[] c = C.getArrayUnsafe();
        ReadAccess.OperandF b = ReadAccess.operand(B, c);
        Blas blas = Matrices.getBlas();
        blas.sgemm(TTrans.NO_TRANS.val(), TTrans.TRANS.val(), C.numRows(), C.numColumns(), cols, alpha, a,
                0, Math.max(1, rows), b.array, b.offset, b.ld, BETA, c, 0, Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF solve(MatrixF B, MatrixF X) {
        Checks.checkSolve(this, B, X);
        // clone before X gets written, X may be this matrix
        if (this.isSquareMatrix()) {
            return lusolve(a.clone(), rows, X, B);
        }
        return qrsolve(a.clone(), rows, cols, X, B);
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

    // A / B = (B^T \ A^T)^T; both transposes are fresh, so LAPACK may overwrite them
    static MatrixF mrdivide(MatrixF A, MatrixF B) {
        Checks.checkSameCols(A, B);
        MatrixF BT = B.transpose();
        MatrixF AT = A.transpose();
        if (BT.isSquareMatrix()) {
            return lusolve(BT.getArrayUnsafe(), BT.numRows(), AT, AT).transpose();
        }
        return qrsolve(BT.getArrayUnsafe(), BT.numRows(), BT.numColumns(),
                Matrices.createF(BT.numColumns(), AT.numColumns()), AT).transpose();
    }

    // work holds a private copy of the n x n matrix and gets overwritten
    static MatrixF lusolve(float[] work, int n, MatrixF X, MatrixF B) {
        // X may already hold the right-hand sides
        if (X != B) {
            X.setInplace(B);
        }
        PlainLapack.sgesv(Matrices.getLapack(), n, B.numColumns(), work, Math.max(1, n), new int[n],
                X.getArrayUnsafe(), Math.max(1, n));
        return X;
    }

    // work holds a private copy of the mm x nn matrix and gets overwritten
    static MatrixF qrsolve(float[] work, int mm, int nn, MatrixF X, MatrixF B) {
        int rhsCount = B.numColumns();

        SimpleMatrixF tmp = new SimpleMatrixF(Math.max(mm, nn), rhsCount);
        B.submatrix(0, 0, mm - 1, rhsCount - 1, tmp, 0, 0);

        PlainLapack.sgels(Matrices.getLapack(), TTrans.NO_TRANS, mm, nn, rhsCount, work, Math.max(1, mm),
                tmp.getArrayUnsafe(), Math.max(1, Math.max(mm, nn)));

        return tmp.submatrix(0, 0, nn - 1, rhsCount - 1, X, 0, 0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixF copy() {
        return new SimpleMatrixF(this);
    }
}
