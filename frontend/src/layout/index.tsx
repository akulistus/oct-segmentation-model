import React from "react";
import Grid from "@mui/material/Grid";
import Upload from "../components/upload";

const Layout: React.FC = () => {
  return(
    <Grid container spacing={5}>
      <Grid size={{md: 4}}>
    
      </Grid>
      <Grid container spacing={2} size={{md: 8}}>
        <Grid size={{xs: 12, md: 6}}>
          <Upload /> 
        </Grid>
        <Grid size={{xs: 12, md: 6}}>
          <Upload /> 
        </Grid>
      </Grid>
    </Grid>
  );
};

export default Layout;