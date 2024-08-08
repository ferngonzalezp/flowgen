"""Created 2023-06-08 by C. Lapeyre (lapeyre@cerfacs.fr)"""
from typing import Optional
from textwrap import dedent
from pathlib import Path


def write_xmf(
    destination: str,
    shape: tuple[int],
    spacing: tuple[int],
    scalars: list[str],
    vectors: Optional[list[tuple[str, int]]] = None,
    verbose: bool = False,
):
    """Write xmf file for 3D structured hdf5 data"""
    XMF_TEMPLATE = dedent(
        """\
    <?xml version="1.0" ?>
    <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
    <Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">
      <Domain>
        <Grid Name="" GridType="Uniform">
          <Topology TopologyType="3DCORECTMesh" Dimensions="{shape[0]} {shape[1]} {shape[2]}"/>
          <Geometry GeometryType="ORIGIN_DXDYDZ">
            <DataItem Name="Origin" Dimensions="3" NumberType="Float" Precision="4" Format="XML">
                            0 0 0
            </DataItem>
            <DataItem Name="Spacing" Dimensions="3" NumberType="Float" Precision="4" Format="XML">
                            {spacing[0]} {spacing[1]} {spacing[2]}
            </DataItem>
          </Geometry>
    {attributes}
        </Grid>
      </Domain>
    </Xdmf>"""
    )

    XMF_SCALAR_ATTR_TEMPLATE = """\
          <Attribute Name="{name}" AttributeType="Scalar" Center="Node">
              <DataItem Format="HDF" NumberType="Float" Precision="4" Dimensions="{shape[0]} {shape[1]} {shape[2]}">{sol_name}:/{name}</DataItem>
          </Attribute>
    """

    XMF_VECTOR_ATTR_TEMPLATE = """\
          <Attribute Name="{name}" AttributeType="Vector" Center="Node">
              <DataItem Format="HDF" NumberType="Float" Precision="4" Dimensions="{shape[0]} {shape[1]} {shape[2]} {shape[3]}">{sol_name}:/{name}</DataItem>
          </Attribute>
    """

    destination = Path(destination)
    if verbose:
        print("\t - Writing", destination)
    with open(destination.parent / f"{destination.stem}.xmf", "w") as xmf:
        attributes = "".join(
            [
                XMF_SCALAR_ATTR_TEMPLATE.format(
                    name=field, shape=shape, sol_name=destination.name
                )
                for field in scalars
            ]
        )
        if vectors is not None:
            attributes += "".join(
                [
                    XMF_VECTOR_ATTR_TEMPLATE.format(
                        name=field, shape=shape + (dims,), sol_name=destination.name
                    )
                    for field, dims in vectors
                ]
            )
        xmf.write(
            XMF_TEMPLATE.format(attributes=attributes, shape=shape, spacing=spacing)
        )
