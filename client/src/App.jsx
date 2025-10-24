import { ThemeProvider } from "@mui/material/styles";
import { theme } from "src/theme";
import CssBaseline from "@mui/material/CssBaseline";
import { Snackbar, PageWrapper } from "src/components";
import AppNavigator from "src/navigation/AppNavigator";

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Snackbar />
      <PageWrapper>
        <AppNavigator />
      </PageWrapper>
    </ThemeProvider>
  );
}

export default App;
