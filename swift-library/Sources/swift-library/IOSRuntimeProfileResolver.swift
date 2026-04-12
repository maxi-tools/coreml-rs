import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Resolves iOS runtime hardware into a stable profile id used by model/profile selection.
///
/// Resolution order:
/// 1. UserDefaults override (`maxi.ios.profile_id`, `maxi_ios_profile_id`, `ios_profile_id`).
/// 2. Environment override (`MAXI_IOS_PROFILE_ID`, `MAXI_PROFILE_ID`, `IOS_PROFILE_ID`) when present.
/// 3. Chip hint mapping (for hosts that expose chip/brand strings).
/// 4. Device/model identifier mapping (including simulator model identifier).
/// 5. Fallback to `ios_a16_min`.
///
/// Supported profile ids:
/// - `ios_a16_min`
/// - `ios_a17_balanced`
/// - `ios_m1_sustained`
/// - `ios_m2_plus`
/// - `apple_m4_peak`
public enum IOSRuntimeProfileResolver {
    public static let defaultProfileID = "ios_a16_min"

    private static let supportedProfileIDs: Set<String> = [
        "ios_a16_min",
        "ios_a17_balanced",
        "ios_m1_sustained",
        "ios_m2_plus",
        "apple_m4_peak",
    ]

    private static let profileAliases: [String: String] = [
        "ios_a14_compat": "ios_a16_min",
        "ios_a15_compat": "ios_a16_min",
        "ios_a16": "ios_a16_min",
        "ios_a17": "ios_a17_balanced",
        "ios_a18": "ios_a17_balanced",
        "ios_m1": "ios_m1_sustained",
        "ios_m2": "ios_m2_plus",
        "ios_m3": "ios_m2_plus",
        "ios_m4": "apple_m4_peak",
    ]

    private static let userDefaultsOverrideKeys = [
        "maxi.ios.profile_id",
        "maxi_ios_profile_id",
        "ios_profile_id",
    ]

    private static let environmentOverrideKeys = [
        "MAXI_IOS_PROFILE_ID",
        "MAXI_PROFILE_ID",
        "IOS_PROFILE_ID",
    ]

    private static let chipHintKeys = [
        "MAXI_IOS_CHIP",
        "MAXI_APPLE_CHIP",
        "MAXI_DEVICE_CHIP",
    ]

    /// Returns a canonical profile id.
    public static func resolveProfileID() -> String {
        if let override = resolveOverrideProfileID() {
            return override
        }

        if let chipHint = resolveChipHint(), let byChip = profileID(forChipHint: chipHint) {
            return byChip
        }

        if let modelIdentifier = resolveModelIdentifier(),
            let byModel = profileID(forModelIdentifier: modelIdentifier)
        {
            return byModel
        }

        return defaultProfileID
    }

    private static func resolveOverrideProfileID() -> String? {
        for key in userDefaultsOverrideKeys {
            if let value = UserDefaults.standard.string(forKey: key),
                let normalized = normalizeProfileID(value)
            {
                return normalized
            }
        }

        let env = ProcessInfo.processInfo.environment
        for key in environmentOverrideKeys {
            if let value = env[key], let normalized = normalizeProfileID(value) {
                return normalized
            }
        }

        return nil
    }

    private static func resolveChipHint() -> String? {
        let env = ProcessInfo.processInfo.environment
        for key in chipHintKeys {
            if let value = env[key], !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return value.lowercased()
            }
        }

        #if os(macOS)
        if let brand = sysctlString(named: "machdep.cpu.brand_string") {
            return brand.lowercased()
        }
        #endif

        return nil
    }

    private static func resolveModelIdentifier() -> String? {
        guard let machine = sysctlString(named: "hw.machine") else {
            return nil
        }

        let normalizedMachine = machine.lowercased()
        let env = ProcessInfo.processInfo.environment

        if normalizedMachine == "x86_64" || normalizedMachine == "arm64" || normalizedMachine == "i386" {
            if let simulatorID = env["SIMULATOR_MODEL_IDENTIFIER"],
                !simulatorID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            {
                return simulatorID.lowercased()
            }
        }

        return normalizedMachine
    }

    private static func normalizeProfileID(_ raw: String) -> String? {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if supportedProfileIDs.contains(normalized) {
            return normalized
        }
        return profileAliases[normalized]
    }

    private static func profileID(forChipHint chipHint: String) -> String? {
        let normalized = chipHint.lowercased()

        if normalized.contains("m4") {
            return "apple_m4_peak"
        }
        if normalized.contains("m3") || normalized.contains("m2") {
            return "ios_m2_plus"
        }
        if normalized.contains("m1") {
            return "ios_m1_sustained"
        }
        if normalized.contains("a18") || normalized.contains("a17") {
            return "ios_a17_balanced"
        }
        if normalized.contains("a16") || normalized.contains("a15") || normalized.contains("a14") {
            return "ios_a16_min"
        }

        return nil
    }

    private static func profileID(forModelIdentifier modelIdentifier: String) -> String? {
        let normalized = modelIdentifier.lowercased()

        if normalized.hasPrefix("iphone") {
            guard let generation = generationNumber(prefix: "iphone", from: normalized) else {
                return "ios_a16_min"
            }
            if generation >= 16 {
                return "ios_a17_balanced"
            }
            return "ios_a16_min"
        }

        if normalized.hasPrefix("ipad") {
            guard let generation = generationNumber(prefix: "ipad", from: normalized) else {
                return "ios_a16_min"
            }
            if generation >= 16 {
                return "apple_m4_peak"
            }
            if generation >= 14 {
                return "ios_m2_plus"
            }
            if generation >= 13 {
                return "ios_m1_sustained"
            }
            return "ios_a16_min"
        }

        if normalized.hasPrefix("mac") {
            if let chipHint = resolveChipHint() {
                return profileID(forChipHint: chipHint)
            }
            return defaultProfileID
        }

        return nil
    }

    private static func generationNumber(prefix: String, from modelIdentifier: String) -> Int? {
        guard modelIdentifier.hasPrefix(prefix) else {
            return nil
        }

        let start = modelIdentifier.index(modelIdentifier.startIndex, offsetBy: prefix.count)
        let suffix = modelIdentifier[start...]
        let digits = suffix.prefix { $0.isNumber }

        guard !digits.isEmpty else {
            return nil
        }

        return Int(digits)
    }

    private static func sysctlString(named name: String) -> String? {
        var size: size_t = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else {
            return nil
        }

        var buffer = [CChar](repeating: 0, count: Int(size))
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else {
            return nil
        }

        return String(cString: buffer)
    }
}
