import InputSection from "./sidebar/InputSection";
import FeaturesSection from "./sidebar/FeaturesSection";
import MaterialsSection from "./sidebar/MaterialsSection";
import VectorsSection from "./sidebar/VectorsSection";
import PerformanceSection from "./sidebar/PerformanceSection";
import ClassificationSection from "./sidebar/ClassificationSection";
import ActionsSection from "./sidebar/ActionsSection";

export default function Sidebar() {
  return (
    <div className="flex flex-col gap-0.5 p-3">
      <InputSection />
      <FeaturesSection />
      <MaterialsSection />
      <VectorsSection />
      <PerformanceSection />
      <ClassificationSection />
      <ActionsSection />
    </div>
  );
}
