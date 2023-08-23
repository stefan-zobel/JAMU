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

import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.Random;

/*
 * 256-bit {@code xoshiro256**} pseudo random generator suggested by
 * <a href=https://arxiv.org/pdf/1805.01407.pdf>David Blackman and Sebastiano
 * Vigna (2019)</a>.
 * <p>
 * This generator has a period of 2<sup>256</sup>&nbsp;&minus;&nbsp;1.
 * <p>
 * This generator is {@code 4}-dimensionally equidistributed.
 */
@SuppressWarnings("serial")
final class XoShiRo256StarStar extends Random {

    private static final double DOUBLE_NORM = 1.0 / (1L << 53);
    // the golden ratio scaled to 64 bits
    private static final long GOLDEN = 0x9e3779b97f4a7c15L;

    private static SecureRandom sr = null;

    private long x0;
    private long x1;
    private long x2;
    private long x3;

    public XoShiRo256StarStar() {
        this(initialSeed());
    }

    public XoShiRo256StarStar(long seed) {
        super(seed);
    }

    @Override
    public void setSeed(long seed) {
        init(seed(seed));
    }

    @Override
    protected final int next(int bits) {
        return (int) (nextLong() >>> (64 - bits));
    }

    public float nextFloat(float min, float max) {
        return min + (max - min) * nextFloat();
    }

    public double nextDouble(double min, double max) {
        return min + (max - min) * nextDouble();
    }

    public long nextLong(long min, long max) {
        return min + nextLong((max - min) + 1L);
    }

    public double nextGaussian(double mean, double stdDeviation) {
        return mean + stdDeviation * nextGaussian();
    }

    public float nextGaussianFloat() {
        return (float) nextGaussian();
    }

    public float nextGaussianFloat(float mean, float stdDeviation) {
        return (float) nextGaussian(mean, stdDeviation);
    }

    @Override
    public double nextDouble() {
        return (nextLong() >>> 11) * DOUBLE_NORM;
    }

    @Override
    public long nextLong() {
        long s1 = x1;
        long t = s1 << 17;
        long x = s1 + (s1 << 2);
        long rnd = ((x << 7) | (x >>> 57));
        rnd += rnd << 3;

        long s2 = (x2 ^= x0);
        long s3 = (x3 ^= s1);
        x1 ^= s2;
        x0 ^= s3;

        x2 ^= t;
        x3 = ((s3 << 45) | (s3 >>> 19));

        return rnd;
    }

    public long nextLong(long n) {
        if (n <= 0L) {
            throw new IllegalArgumentException("n must be positive");
        }
        final long nMinus1 = n - 1L;
        long x = nextLong();
        if ((n & nMinus1) == 0L) {
            // power of two shortcut
            return x & nMinus1;
        }
        // rejection-based algorithm to get uniform longs
        for (long y = x >>> 1; y + nMinus1 - (x = y % n) < 0L; y = nextLong() >>> 1) {
            ;
        }
        return x;
    }

    private void init(long nextSeed) {
        SecureRandom rnd = getSecureRandom();
        rnd.setSeed(nextSeed);
        x0 = rnd.nextLong();
        x1 = rnd.nextLong();
        x2 = rnd.nextLong();
        x3 = rnd.nextLong();
        // escape
        for (int i = 0; i < 20; ++i) {
            nextLong();
        }
    }

    /*
     * Computes a deterministic seed value from a given value.
     * 
     * @param seed the seed to start with
     * 
     * @return a deterministically computed seed value
     */
    private static long seed(long seed) {
        seed = (seed == 0L) ? -1L : seed;
        return rrxmrrxmsx(seed + GOLDEN);
    }

    /*
     * Pelle Evensen's even better mixer. See
     * "https://mostlymangling.blogspot.com/2019/01/better-stronger-mixer-and-test-procedure.html".
     * 
     * @param v long to mix
     * 
     * @return the mixed long
     */
    private static long rrxmrrxmsx(long v) {
        v ^= ((v >>> 25) | (v << 39)) ^ ((v >>> 50) | (v << 14));
        v *= 0xa24baed4963ee407L;
        v ^= ((v >>> 24) | (v << 40)) ^ ((v >>> 49) | (v << 15));
        v *= 0x9fb21c651e98df25L;
        return v ^ (v >>> 28);
    }

    private static long initialSeed() {
        byte[] seedBytes = getSecureRandom().generateSeed(8);
        long s = (long) (seedBytes[0]) & 0xffL;
        for (int i = 1; i < 8; ++i) {
            s = (s << 8) | ((long) (seedBytes[i]) & 0xffL);
        }
        return s;
    }

    private static SecureRandom getSecureRandom() {
        if (sr == null) {
            try {
                sr = SecureRandom.getInstance("SHA1PRNG");
            } catch (NoSuchAlgorithmException e) {
                throw new UnsupportedOperationException("SHA1PRNG", e);
            }
        }
        return sr;
    }
}
