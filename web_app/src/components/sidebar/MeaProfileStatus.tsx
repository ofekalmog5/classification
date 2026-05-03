import { useState, useEffect } from "react";
import { getMeaProfileStatus } from "../../api/client";

interface Status {
  source: string;
  name: string;
  material_count: number;
  is_factory_default: boolean;
  profile_path: string;
}

export default function MeaProfileStatus() {
  const [status, setStatus] = useState<Status | null>(null);

  useEffect(() => {
    getMeaProfileStatus()
      .then(setStatus)
      .catch(() => {});
  }, []);

  return (
    <details className="group">
      <summary className="flex items-center cursor-pointer select-none py-2 px-1 text-xs font-semibold uppercase tracking-wider text-surface-400 hover:text-surface-200 transition-colors">
        <svg className="w-3 h-3 mr-1.5 transition-transform group-open:rotate-90" fill="currentColor" viewBox="0 0 20 20">
          <path d="M6 6l8 4-8 4V6z" />
        </svg>
        MEA Profile
        {status && !status.is_factory_default && (
          <span className="ml-2 px-1.5 py-0.5 rounded text-[9px] font-bold bg-green-800 text-green-200 uppercase">
            Custom
          </span>
        )}
      </summary>

      <div className="pb-3 px-1 space-y-1">
        {!status ? (
          <p className="text-[10px] text-surface-600 px-1">Loading…</p>
        ) : (
          <div className="text-[10px] text-surface-400 space-y-0.5 px-1">
            <p className="font-semibold text-surface-200">{status.name}</p>
            <p>{status.material_count} material{status.material_count !== 1 ? "s" : ""}</p>
            {status.is_factory_default ? (
              <p className="text-surface-600">Factory defaults — calibrate with MEA Calibration Tool</p>
            ) : (
              <p className="text-surface-500 truncate" title={status.profile_path}>
                {status.profile_path.split(/[\\/]/).pop()}
              </p>
            )}
          </div>
        )}
      </div>
    </details>
  );
}
