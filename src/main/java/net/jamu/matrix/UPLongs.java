/*
 * Copyright 2021 Stefan Zobel
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
 * Upper triangular square matrix of longs. Indexing is zero-based.
 */
final class UPLongs extends UpperPacked {

    private final long[] a;

    UPLongs(int dim) {
        super(dim);
        a = new long[length()];
    }

    long get(int row, int col) {
        return a[idx(row, col)];
    }

    void set(int row, int col, long val) {
        a[idx(row, col)] = val;
    }

    // for toString() only
    Object getVal(int row, int col) {
        return get(row, col);
    }
}
