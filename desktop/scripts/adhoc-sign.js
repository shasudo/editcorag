"use strict";

/**
 * Re-seal the macOS bundle with an ad-hoc signature, after packaging.
 *
 * electron-builder signs nothing when no Developer ID is configured, but the
 * Electron binary arrives already signed — and dropping a 1.5 GB Python runtime
 * into Resources/ invalidates that seal. `codesign --verify` then fails with
 * "code has no resources but signature indicates they must be present", which
 * is what surfaces to a user as "the app is damaged and can't be opened".
 *
 * Ad-hoc signing needs no Apple account and no payment. It is also not
 * optional: arm64 refuses to execute an unsigned binary at all. What it does
 * not provide is notarization, which only matters for apps downloaded from the
 * web — an app copied from a USB stick is never quarantined, so Gatekeeper
 * never assesses it.
 */

const { execFileSync } = require("node:child_process");
const path = require("node:path");

exports.default = async function afterPack(context) {
  if (context.electronPlatformName !== "darwin") return;

  const app = path.join(
    context.appOutDir,
    `${context.packager.appInfo.productFilename}.app`,
  );

  execFileSync("codesign", ["--force", "--deep", "--sign", "-", app], {
    stdio: "inherit",
  });

  // Refuse to ship a bundle whose seal does not verify: the failure mode is
  // silent until a user double-clicks it.
  execFileSync("codesign", ["--verify", "--deep", "--strict", app], {
    stdio: "inherit",
  });
  console.log("  • ad-hoc signature applied and verified");
};
