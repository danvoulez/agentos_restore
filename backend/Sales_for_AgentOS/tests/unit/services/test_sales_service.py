import pytest, asyncio
from app.services.sales_service import SalesService, CreateSaleInput, CreateSaleItemInput
from app.db.repositories.sale_repository import SaleRepository
from app.db.repositories.product_repository import ProductRepository
from app.services.product_service import ProductService

@pytest.mark.asyncio
async def test_dummy_create_sale(mongo_mock):
    product_repo = ProductRepository(get_database())
    sale_repo = SaleRepository(get_database())
    product_service = ProductService(product_repo)

    service = SalesService(sale_repo, product_service, mongo_mock)

    sale_input = CreateSaleInput(
        client_id="c1", agent_id="a1",
        items=[CreateSaleItemInput(sku="SKU", quantity=1)]
    )
    with pytest.raises(Exception):
        await service.create_sale(sale_input)