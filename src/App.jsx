import { ThemeProvider } from "@mui/material/styles";
import { theme } from "src/theme";
import { Toaster, PageWrapper } from "src/components";
import AppNavigator from "src/navigation/AppNavigator";

function App() {
  return (
    <ThemeProvider theme={theme}>
      <PageWrapper>
        <AppNavigator />
      </PageWrapper>
      <Toaster />
    </ThemeProvider>
  );
}

export default App;
