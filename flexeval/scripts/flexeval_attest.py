"""Command line tool to create or verify a post-quantum attestation for results.

After an evaluation run has written its result files, this tool signs them with
ML-DSA (FIPS 204) so the published results are tamper-evident and, when the
signer's public key is pinned, verifiably authentic.
"""

from __future__ import annotations

import argparse
import base64
import sys

from loguru import logger

from flexeval.core.attestation import (
    DEFAULT_ALGORITHM,
    SUPPORTED_ALGORITHMS,
    create_attestation,
    generate_key_pair,
    verify_attestation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or verify a post-quantum attestation for FlexEval results.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    attest_parser = subparsers.add_parser("attest", help="Attest the result files in a directory.")
    attest_parser.add_argument("output_dir", help="Directory holding the run result files.")
    attest_parser.add_argument("--algorithm", default=DEFAULT_ALGORITHM, choices=SUPPORTED_ALGORITHMS)
    attest_parser.add_argument(
        "--public-key-out",
        default=None,
        help="Write the signer public key (base64) to this file so others can verify authenticity.",
    )

    verify_parser = subparsers.add_parser("verify", help="Verify the attestation in a directory.")
    verify_parser.add_argument("output_dir", help="Directory holding the attestation and result files.")
    verify_parser.add_argument(
        "--public-key",
        default=None,
        help="Path to a base64 public key file used to pin the expected signer.",
    )

    args = parser.parse_args()

    if args.command == "attest":
        key_pair = None
        if args.public_key_out is not None:
            key_pair = generate_key_pair(args.algorithm)
        attestation = create_attestation(args.output_dir, algorithm=args.algorithm, key_pair=key_pair)
        if args.public_key_out is not None:
            with open(args.public_key_out, "w") as f:
                f.write(str(attestation["public_key"]))
        logger.info(f"Wrote attestation with {attestation['algorithm']} over {len(attestation['files'])} file(s).")
    elif args.command == "verify":
        expected_public_key = None
        if args.public_key is not None:
            with open(args.public_key) as f:
                expected_public_key = base64.b64decode(f.read().strip())
        result = verify_attestation(args.output_dir, expected_public_key=expected_public_key)
        logger.info(result.reason)
        if not result.valid:
            sys.exit(1)


if __name__ == "__main__":
    main()
